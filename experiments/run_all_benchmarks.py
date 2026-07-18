from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.integrate import odeint
from scipy.stats import f as f_distribution
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPL_CONFIG_DIR = ROOT / "experiments" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

from xneural_var import (  # noqa: E402
    CMLPTrainingConfig,
    GVARBaselineTrainingConfig,
    GVARTrainingConfig,
    construct_lagged_dataset,
    fit_cmlp,
    fit_gvar,
    fit_gvar_ngc,
    plot_causal_gate_by_lag,
    plot_edge_lag_boxplots,
)


EXPERIMENT_VERSION = 4
OUT_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures"
RAW_DIR = OUT_DIR / "raw_benchmark_v2"
RESULT_JSON = OUT_DIR / "benchmark_results.json"
METHOD_CSV = OUT_DIR / "method_comparison.csv"
LAMBDA_CSV = OUT_DIR / "xneural_lambda_path.csv"
NOTEBOOK_PATH = OUT_DIR / "xneural_var_benchmarks.ipynb"


# Lorenz-96 follows the published Neural-GC benchmark at T=500. The original
# paper also reports T=250 and T=1000; T=500 is the predeclared condition used
# here for a tractable four-method comparison.
LORENZ_CONFIG: dict[str, Any] = {
    "p": 20,
    "order": 5,
    "T": 500,
    "forcing_values": [10.0, 40.0],
    "delta_t": 0.05,
    "observation_noise_sd": 0.1,
    "burn_in": 1000,
    "data_seed": 0,
    "calibration_seed": 1000,
    "evaluation_seeds": [0, 1, 2, 3, 4],
    "hidden_layer_size": 100,
    "num_hidden_layers": 1,
    "max_epochs": 150,
    "batch_size": 10000,
    "ista_learning_rate": 5e-2,
    "gvar_learning_rate": 1e-3,
    "ridge_lambda": 1e-2,
    "lambda_smooth_xneural": 1e-2,
    "coefficient_weight_decay": 1e-4,
}

LORENZ_XNEURAL_LAMBDAS = [0.02, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.28]
LORENZ_CMLP_LAMBDAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]
LORENZ_GVAR_LAMBDAS = np.linspace(0.0, 3.0, 5).tolist()
LORENZ_GVAR_GAMMAS = np.linspace(0.0, 0.025, 5).tolist()


# A deliberately nontrivial sign-recovery sanity check: the fitted order is
# over-specified, effects have mixed signs and strengths, innovations are
# contemporaneously correlated, and the sample size is moderate.
LINEAR_CONFIG: dict[str, Any] = {
    "p": 10,
    "true_order": 1,
    "fit_order": 5,
    "T": 300,
    "burn_in": 500,
    "noise_sd": 0.5,
    "innovation_correlation": 0.15,
    "calibration_seed": 1000,
    "evaluation_seeds": [0, 1, 2, 3, 4],
    "hidden_layer_size": 32,
    "num_hidden_layers": 1,
    "max_epochs": 200,
    "batch_size": 10000,
    "ista_learning_rate": 5e-2,
    "gvar_learning_rate": 2e-3,
    "ridge_lambda": 1e-2,
    "lambda_smooth_xneural": 0.0,
    "coefficient_weight_decay": 1e-4,
    "fdr_level": 0.05,
}

LINEAR_XNEURAL_LAMBDAS = [0.01, 0.04, 0.07, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
LINEAR_CMLP_LAMBDAS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
LINEAR_GVAR_LAMBDAS = [0.0, 0.05, 0.2, 0.5]
LINEAR_GVAR_GAMMAS = [0.0, 0.005, 0.02]


def lorenz96_rhs(x: np.ndarray, _time: float, forcing: float) -> np.ndarray:
    p = len(x)
    dxdt = np.empty(p, dtype=np.float64)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + forcing
    return dxdt


def simulate_lorenz96_ngc(
    *,
    p: int,
    T: int,
    forcing: float,
    delta_t: float,
    observation_noise_sd: float,
    burn_in: int,
    seed: int,
) -> np.ndarray:
    """Neural-GC author's Lorenz-96 generator, including observation noise."""
    rng = np.random.RandomState(seed)
    x0 = rng.normal(scale=0.01, size=p)
    times = np.linspace(0.0, (T + burn_in) * delta_t, T + burn_in)
    data = odeint(lorenz96_rhs, x0, times, args=(forcing,))
    data += rng.normal(scale=observation_noise_sd, size=(T + burn_in, p))
    return data[burn_in:].astype(np.float32)


def true_lorenz96_graph(p: int) -> np.ndarray:
    graph = np.zeros((p, p), dtype=np.int64)
    for target in range(p):
        for source in (target, (target + 1) % p, (target - 1) % p, (target - 2) % p):
            graph[target, source] = 1
    return graph


def standardize_full(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    return ((data - mean) / np.where(std < 1e-8, 1.0, std)).astype(np.float32)


def _companion_radius(coeffs: np.ndarray) -> float:
    order, p, _ = coeffs.shape
    top = np.hstack([coeffs[lag] for lag in range(order)])
    if order == 1:
        companion = top
    else:
        bottom = np.hstack([np.eye(p * (order - 1)), np.zeros((p * (order - 1), p))])
        companion = np.vstack([top, bottom])
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def true_linear_var_coefficients(p: int = 10, order: int = 1) -> np.ndarray:
    if p != 10 or order != 1:
        raise ValueError("the predeclared linear benchmark uses p=10 and order=1")
    coeffs = np.zeros((order, p, p), dtype=np.float64)
    coeffs[0, np.arange(p), np.arange(p)] = 0.24

    for target in range(p):
        source_strong = (target - 1) % p
        source_weak = (target + 2) % p
        coeffs[0, target, source_strong] = (0.34 if target % 2 == 0 else -0.34)
        coeffs[0, target, source_weak] = (-0.20 if target % 2 == 0 else 0.20)

    while _companion_radius(coeffs) >= 0.92:
        coeffs *= 0.95
    return coeffs


def simulate_linear_var(
    *,
    coeffs: np.ndarray,
    T: int,
    burn_in: int,
    noise_sd: float,
    innovation_correlation: float,
    seed: int,
) -> np.ndarray:
    order, p, _ = coeffs.shape
    rng = np.random.default_rng(seed)
    corr = (1.0 - innovation_correlation) * np.eye(p) + innovation_correlation * np.ones((p, p))
    covariance = noise_sd**2 * corr
    total = burn_in + T + order
    innovations = rng.multivariate_normal(np.zeros(p), covariance, size=total)
    data = np.zeros((total, p), dtype=np.float64)
    data[:order] = innovations[:order]
    for time_idx in range(order, total):
        value = np.zeros(p, dtype=np.float64)
        for lag in range(1, order + 1):
            value += coeffs[lag - 1] @ data[time_idx - lag]
        data[time_idx] = value + innovations[time_idx]
    return data[burn_in + order : burn_in + order + T].astype(np.float32)


def _evaluation_mask(p: int, include_diagonal: bool) -> np.ndarray:
    return np.ones((p, p), dtype=bool) if include_diagonal else ~np.eye(p, dtype=bool)


def binary_metrics(pred: np.ndarray, truth: np.ndarray, *, include_diagonal: bool = False) -> dict[str, float]:
    mask = _evaluation_mask(truth.shape[0], include_diagonal)
    pred_b = np.asarray(pred[mask], dtype=bool)
    truth_b = np.asarray(truth[mask], dtype=bool)
    tp = int(np.logical_and(pred_b, truth_b).sum())
    fp = int(np.logical_and(pred_b, ~truth_b).sum())
    fn = int(np.logical_and(~pred_b, truth_b).sum())
    tn = int(np.logical_and(~pred_b, ~truth_b).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "selected_edges": float(pred_b.sum()),
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    sum_pos = float(ranks[labels == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), labels.size - 1]
    tp_at_threshold = tp[distinct_ends]
    precision = tp_at_threshold / (distinct_ends + 1)
    previous_tp = np.r_[0, tp_at_threshold[:-1]]
    recall_increment = (tp_at_threshold - previous_tp) / positives
    return float(np.sum(precision * recall_increment))


def score_metrics(score: np.ndarray, truth: np.ndarray, *, include_diagonal: bool = False) -> dict[str, float]:
    mask = _evaluation_mask(truth.shape[0], include_diagonal)
    labels = truth[mask].astype(np.int64)
    scores = np.asarray(score[mask], dtype=np.float64)
    return {"auroc": auroc(scores, labels), "auprc": average_precision(scores, labels)}


def path_survival_score(rows: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Convert an exact-zero regularization path to an edge ranking.

    An edge receives the largest regularization strength at which it remains
    nonzero. For a nested path this is exactly the ROC/PR curve obtained by
    sweeping lambda. Violations of nesting are counted and reported.
    """
    ordered = sorted(rows, key=lambda row: row["lambda"])
    graphs = [np.asarray(row["graph"], dtype=np.int64) for row in ordered]
    score = np.zeros_like(graphs[0], dtype=np.float64)
    violations = 0
    previous = graphs[0].astype(bool)
    for row, graph in zip(ordered, graphs):
        selected = graph.astype(bool)
        score[selected] = np.maximum(score[selected], float(row["lambda"]))
        violations += int(np.logical_and(selected, ~previous).sum())
        previous = selected
    return score, violations


def benjamini_hochberg(pvalues: np.ndarray, q: float, mask: np.ndarray) -> np.ndarray:
    selected = np.zeros_like(pvalues, dtype=bool)
    flat = pvalues[mask]
    order = np.argsort(flat)
    ranked = flat[order]
    limits = q * np.arange(1, ranked.size + 1) / ranked.size
    passing = np.flatnonzero(ranked <= limits)
    if passing.size:
        cutoff = ranked[passing[-1]]
        selected[mask] = flat <= cutoff
    return selected


def fit_linear_var_with_tests(data: np.ndarray, order: int, fdr_level: float) -> dict[str, Any]:
    dataset = construct_lagged_dataset(data, order)
    predictors = dataset.predictors.reshape(dataset.predictors.shape[0], -1).astype(np.float64)
    responses = dataset.responses.astype(np.float64)
    n_samples, p = responses.shape
    design = np.column_stack([np.ones(n_samples), predictors])
    beta, _, _, _ = np.linalg.lstsq(design, responses, rcond=None)
    fitted = design @ beta
    residuals = responses - fitted
    rss_full = np.sum(residuals**2, axis=0)
    df_denominator = n_samples - design.shape[1]
    if df_denominator <= 0:
        raise ValueError("not enough observations for the requested VAR order")

    # The partial F statistic for removing all lags of one source can be
    # calculated from the unrestricted fit. This is algebraically equivalent
    # to refitting every restricted VAR, but avoids p^2 least-squares solves.
    xtx_inverse = np.linalg.pinv(design.T @ design)
    statistics = np.zeros((p, p), dtype=np.float64)
    denominator = np.maximum(rss_full / df_denominator, np.finfo(float).tiny)
    for source in range(p):
        indices = np.array([1 + lag_idx * p + source for lag_idx in range(order)])
        source_beta = beta[indices, :]
        source_covariance = xtx_inverse[np.ix_(indices, indices)]
        solved = np.linalg.solve(source_covariance, source_beta)
        extra_sum_squares = np.sum(source_beta * solved, axis=0)
        statistics[:, source] = np.maximum(extra_sum_squares, 0.0) / order / denominator
    pvalues = f_distribution.sf(statistics, order, df_denominator)

    offdiag = ~np.eye(p, dtype=bool)
    graph = benjamini_hochberg(pvalues, q=fdr_level, mask=offdiag).astype(np.int64)
    coeffs_model_order = beta[1:].reshape(order, p, p).transpose(0, 2, 1)
    coeffs_actual_order = coeffs_model_order[::-1]
    return {
        "graph": graph,
        "score": -np.log10(np.clip(pvalues, np.finfo(float).tiny, 1.0)),
        "pvalues": pvalues,
        "f_statistics": statistics,
        "coeffs_actual_order": coeffs_actual_order,
        "train_mse": float(np.mean(residuals**2)),
    }


def coefficient_sign_accuracy(estimated_actual_order: np.ndarray, true_actual_order: np.ndarray) -> float:
    max_order = min(estimated_actual_order.shape[0], true_actual_order.shape[0])
    estimated = estimated_actual_order[:max_order]
    truth = true_actual_order[:max_order]
    offdiag = ~np.eye(truth.shape[1], dtype=bool)
    mask = (np.abs(truth) > 1e-12) & offdiag[None, :, :]
    return float(np.mean(np.sign(estimated[mask]) == np.sign(truth[mask])))


def _effective_coeff_sign(result: Any, true_coeffs: np.ndarray) -> float:
    median_model_order = np.median(np.asarray(result.coeffs), axis=0)
    return coefficient_sign_accuracy(median_model_order[::-1], true_coeffs)


def _fit_xneural(data: np.ndarray, config: dict[str, Any], lam: float, seed: int, device: str) -> tuple[dict[str, Any], Any]:
    fit_config = GVARTrainingConfig(
        order=int(config["order"]),
        hidden_layer_size=int(config["hidden_layer_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        max_epochs=int(config["max_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["ista_learning_rate"]),
        lambda_ngc=float(lam),
        lambda_smooth=float(config["lambda_smooth_xneural"]),
        coefficient_weight_decay=float(config["coefficient_weight_decay"]),
        regularizer="hierarchical_group_lasso",
        optimizer="ista",
        gate_init=1.0,
        causal_threshold=0.0,
        strength_aggregation="max",
        seed=seed,
        shuffle=False,
        verbose=0,
        device=device,
    )
    start = time.perf_counter()
    result = fit_gvar_ngc(data, fit_config)
    elapsed = time.perf_counter() - start
    gate_norms = result.model.gate_group_norms().detach().cpu().numpy()
    row = {
        "method": "Xneural VAR",
        "lambda": float(lam),
        "seed": int(seed),
        "graph": (gate_norms > 0.0).astype(np.int64).tolist(),
        "score": gate_norms.tolist(),
        "train_mse": float(result.history["mse"][-1]),
        "elapsed_sec": float(elapsed),
    }
    return row, result


def _fit_cmlp(data: np.ndarray, config: dict[str, Any], lam: float, seed: int, device: str) -> tuple[dict[str, Any], Any]:
    fit_config = CMLPTrainingConfig(
        order=int(config["order"]),
        hidden_layer_size=int(config["hidden_layer_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        max_epochs=int(config["max_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["ista_learning_rate"]),
        lambda_ngc=float(lam),
        ridge_lambda=float(config["ridge_lambda"]),
        regularizer="hierarchical_group_lasso",
        optimizer="ista",
        causal_threshold=0.0,
        seed=seed,
        shuffle=False,
        verbose=0,
        device=device,
    )
    start = time.perf_counter()
    result = fit_cmlp(data, fit_config)
    elapsed = time.perf_counter() - start
    strengths = np.asarray(result.causal_strength, dtype=np.float64)
    row = {
        "method": "NGC (cMLP)",
        "lambda": float(lam),
        "seed": int(seed),
        "graph": (strengths > 0.0).astype(np.int64).tolist(),
        "score": strengths.tolist(),
        "train_mse": float(result.history["mse"][-1]),
        "elapsed_sec": float(elapsed),
    }
    return row, result


def _gvar_config(config: dict[str, Any], lam: float, gamma: float, seed: int, device: str) -> GVARBaselineTrainingConfig:
    return GVARBaselineTrainingConfig(
        order=int(config["order"]),
        hidden_layer_size=int(config["hidden_layer_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        max_epochs=int(config["max_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["gvar_learning_rate"]),
        lambda_coeff=float(lam),
        elastic_net_alpha=0.5,
        lambda_smooth=float(gamma),
        coefficient_weight_decay=0.0,
        causal_threshold=0.0,
        strength_aggregation="max",
        seed=seed,
        shuffle=False,
        verbose=0,
        device=device,
    )


def stability_graph(strength: np.ndarray, reversed_strength: np.ndarray, q_levels: int = 20) -> tuple[np.ndarray, float, float]:
    reversed_transposed = reversed_strength.T
    p = strength.shape[0]
    offdiag = ~np.eye(p, dtype=bool)
    alphas = np.linspace(0.0, 1.0, q_levels)
    agreements = []
    for alpha in alphas:
        threshold_forward = np.quantile(strength, alpha)
        threshold_reverse = np.quantile(reversed_transposed, alpha)
        graph_forward = (strength >= threshold_forward).astype(np.int64)
        graph_reverse = (reversed_transposed >= threshold_reverse).astype(np.int64)
        agreement = binary_metrics(graph_forward, graph_reverse, include_diagonal=False)["balanced_accuracy"]
        if graph_forward.sum() <= p or graph_reverse.sum() <= p:
            agreement = 0.0
        if graph_forward.sum() == p**2 or graph_reverse.sum() == p**2:
            agreement = 0.0
        agreements.append(agreement)
    best_idx = int(np.argmax(agreements))
    alpha = float(alphas[best_idx])
    threshold = float(np.quantile(strength, alpha))
    graph = (strength >= threshold).astype(np.int64)
    return graph, alpha, float(agreements[best_idx])


def _fit_gvar(
    data: np.ndarray,
    config: dict[str, Any],
    lam: float,
    gamma: float,
    seed: int,
    device: str,
    stability: bool,
) -> tuple[dict[str, Any], Any]:
    start = time.perf_counter()
    result = fit_gvar(data, _gvar_config(config, lam, gamma, seed, device))
    strength = np.asarray(result.causal_strength, dtype=np.float64)
    graph = np.ones_like(strength, dtype=np.int64)
    alpha = float("nan")
    agreement = float("nan")
    if stability:
        reverse_result = fit_gvar(data[::-1].copy(), _gvar_config(config, lam, gamma, seed, device))
        graph, alpha, agreement = stability_graph(
            strength,
            np.asarray(reverse_result.causal_strength, dtype=np.float64),
        )
    elapsed = time.perf_counter() - start
    row = {
        "method": "GVAR",
        "lambda": float(lam),
        "gamma": float(gamma),
        "seed": int(seed),
        "graph": graph.tolist(),
        "score": strength.tolist(),
        "stability_alpha": alpha,
        "stability_agreement": agreement,
        "train_mse": float(result.history["mse"][-1]),
        "elapsed_sec": float(elapsed),
    }
    return row, result


def _cache_name(job: dict[str, Any]) -> str:
    method = job["method"].replace(" ", "_").replace("(", "").replace(")", "")
    if job["method"] == "cmlp":
        method += "_author_objective"
    parts = [method, f"seed_{job['seed']}"]
    if "lambda_value" in job:
        parts.append(f"lambda_{job['lambda_value']:g}")
    if "gamma" in job:
        parts.append(f"gamma_{job['gamma']:g}")
    if job.get("stability"):
        parts.append("stability")
    return "_".join(parts) + ".json"


def _fit_job(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(job.get("torch_threads", 1)))
    method = job["method"]
    data = np.asarray(job["data"], dtype=np.float32)
    if method == "xneural":
        row, result = _fit_xneural(data, job["config"], job["lambda_value"], job["seed"], job["device"])
    elif method == "cmlp":
        row, result = _fit_cmlp(data, job["config"], job["lambda_value"], job["seed"], job["device"])
    elif method == "gvar":
        row, result = _fit_gvar(
            data,
            job["config"],
            job["lambda_value"],
            job["gamma"],
            job["seed"],
            job["device"],
            bool(job.get("stability", False)),
        )
    else:
        raise ValueError(f"unsupported job method: {method}")

    true_coeffs = job.get("true_coeffs")
    if true_coeffs is not None and method in {"xneural", "gvar"}:
        row["sign_accuracy"] = _effective_coeff_sign(result, np.asarray(true_coeffs, dtype=np.float64))
    return row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cached_jobs(
    jobs: list[dict[str, Any]],
    *,
    cache_dir: Path,
    force: bool,
    workers: int,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    for job in jobs:
        path = cache_dir / _cache_name(job)
        if path.exists() and not force:
            rows.append(read_json(path))
        else:
            pending.append((job, path))
    if not pending:
        return rows

    print(f"Running {len(pending)} fits ({workers} worker(s)) in {cache_dir.name}...")
    if workers <= 1:
        for index, (job, path) in enumerate(pending, start=1):
            row = _fit_job(job)
            write_json(path, row)
            rows.append(row)
            print(f"  [{index:>3}/{len(pending)}] {_cache_name(job)} edges={np.sum(row['graph']):.0f}")
        return rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fit_job, job): (job, path) for job, path in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            job, path = futures[future]
            row = future.result()
            write_json(path, row)
            rows.append(row)
            print(f"  [{index:>3}/{len(pending)}] {_cache_name(job)} edges={np.sum(row['graph']):.0f}")
    return rows


def choose_lambda(rows: list[dict[str, Any]], truth: np.ndarray) -> float:
    ranked = []
    for row in rows:
        metrics = binary_metrics(np.asarray(row["graph"]), truth)
        ranked.append((metrics["f1"], metrics["balanced_accuracy"], -metrics["selected_edges"], row["lambda"]))
    return float(max(ranked)[-1])


def choose_gvar_params(rows: list[dict[str, Any]], truth: np.ndarray) -> tuple[float, float]:
    ranked = []
    for row in rows:
        metrics = score_metrics(np.asarray(row["score"]), truth)
        ranked.append((metrics["auprc"], metrics["auroc"], -row["lambda"], -row["gamma"], row))
    best = max(ranked, key=lambda item: item[:4])[-1]
    return float(best["lambda"]), float(best["gamma"])


def mean_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    critical = float(student_t.ppf(0.975, array.size - 1)) if array.size > 1 else 0.0
    ci95 = critical * std / math.sqrt(array.size) if array.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci95), "n": int(array.size)}


def aggregate_method_runs(method: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "selected_edges",
        "train_mse",
        "elapsed_sec",
    ]
    summary = {name: mean_summary(run[name] for run in runs) for name in metric_names}
    sign_values = [run["sign_accuracy"] for run in runs if "sign_accuracy" in run]
    if sign_values:
        summary["sign_accuracy"] = mean_summary(sign_values)
    return {"method": method, "summary": summary, "runs": runs}


def _path_runs(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    evaluation_seeds: list[int],
    selected_lambda: float,
) -> list[dict[str, Any]]:
    runs = []
    for seed in evaluation_seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        path_score, violations = path_survival_score(seed_rows)
        selected_row = min(seed_rows, key=lambda row: abs(float(row["lambda"]) - selected_lambda))
        run = {
            **score_metrics(path_score, truth, include_diagonal=False),
            **binary_metrics(np.asarray(selected_row["graph"]), truth, include_diagonal=False),
            "auroc_all": score_metrics(path_score, truth, include_diagonal=True)["auroc"],
            "auprc_all": score_metrics(path_score, truth, include_diagonal=True)["auprc"],
            "path_nesting_violations": float(violations),
            "train_mse": float(selected_row["train_mse"]),
            "elapsed_sec": float(sum(row["elapsed_sec"] for row in seed_rows)),
        }
        if "sign_accuracy" in selected_row:
            run["sign_accuracy"] = float(selected_row["sign_accuracy"])
        runs.append(run)
    return runs


def _gvar_runs(rows: list[dict[str, Any]], truth: np.ndarray) -> list[dict[str, Any]]:
    runs = []
    for row in sorted(rows, key=lambda item: item["seed"]):
        run = {
            **score_metrics(np.asarray(row["score"]), truth, include_diagonal=False),
            **binary_metrics(np.asarray(row["graph"]), truth, include_diagonal=False),
            "auroc_all": score_metrics(np.asarray(row["score"]), truth, include_diagonal=True)["auroc"],
            "auprc_all": score_metrics(np.asarray(row["score"]), truth, include_diagonal=True)["auprc"],
            "train_mse": float(row["train_mse"]),
            "elapsed_sec": float(row["elapsed_sec"]),
            "stability_alpha": float(row["stability_alpha"]),
            "stability_agreement": float(row["stability_agreement"]),
        }
        if "sign_accuracy" in row:
            run["sign_accuracy"] = float(row["sign_accuracy"])
        runs.append(run)
    return runs


def _linear_runs(
    datasets: dict[int, np.ndarray],
    truth: np.ndarray,
    order: int,
    fdr_level: float,
    true_coeffs: np.ndarray | None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    runs = []
    details = {}
    for seed, data in datasets.items():
        start = time.perf_counter()
        result = fit_linear_var_with_tests(data, order=order, fdr_level=fdr_level)
        elapsed = time.perf_counter() - start
        run = {
            **score_metrics(result["score"], truth, include_diagonal=False),
            **binary_metrics(result["graph"], truth, include_diagonal=False),
            "auroc_all": score_metrics(result["score"], truth, include_diagonal=True)["auroc"],
            "auprc_all": score_metrics(result["score"], truth, include_diagonal=True)["auprc"],
            "train_mse": float(result["train_mse"]),
            "elapsed_sec": float(elapsed),
        }
        if true_coeffs is not None:
            padded_truth = np.zeros((order, truth.shape[0], truth.shape[1]), dtype=np.float64)
            padded_truth[: true_coeffs.shape[0]] = true_coeffs
            run["sign_accuracy"] = coefficient_sign_accuracy(result["coeffs_actual_order"], padded_truth)
        runs.append(run)
        details[seed] = result
    return runs, details


def lambda_path_summary(rows: list[dict[str, Any]], truth: np.ndarray, evaluation_seeds: list[int]) -> list[dict[str, Any]]:
    summary = []
    for lam in sorted({float(row["lambda"]) for row in rows}):
        lam_rows = [row for row in rows if float(row["lambda"]) == lam and int(row["seed"]) in evaluation_seeds]
        metrics = [binary_metrics(np.asarray(row["graph"]), truth) for row in lam_rows]
        summary.append(
            {
                "lambda": lam,
                "selected_edges": mean_summary(metric["selected_edges"] for metric in metrics),
                "precision": mean_summary(metric["precision"] for metric in metrics),
                "recall": mean_summary(metric["recall"] for metric in metrics),
                "f1": mean_summary(metric["f1"] for metric in metrics),
                "balanced_accuracy": mean_summary(metric["balanced_accuracy"] for metric in metrics),
            }
        )
    return summary


def _job(
    method: str,
    data: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: str,
    workers: int,
    **kwargs: Any,
) -> dict[str, Any]:
    job = {
        "method": method,
        "data": data,
        "config": config,
        "seed": seed,
        "device": device,
        "torch_threads": max(1, (os.cpu_count() or 1) // max(workers, 1)),
    }
    job.update(kwargs)
    return job


def run_lorenz_condition(
    forcing: float,
    *,
    force: bool,
    workers: int,
    device: str,
) -> dict[str, Any]:
    config = dict(LORENZ_CONFIG)
    config["order"] = config.pop("order")
    data = standardize_full(
        simulate_lorenz96_ngc(
            p=config["p"],
            T=config["T"],
            forcing=forcing,
            delta_t=config["delta_t"],
            observation_noise_sd=config["observation_noise_sd"],
            burn_in=config["burn_in"],
            seed=config["data_seed"],
        )
    )
    truth = true_lorenz96_graph(config["p"])
    cache = RAW_DIR / f"lorenz_f{forcing:g}"

    calibration_jobs = []
    for lam in LORENZ_XNEURAL_LAMBDAS:
        calibration_jobs.append(_job("xneural", data, config, config["calibration_seed"], device, workers, lambda_value=lam))
    for lam in LORENZ_CMLP_LAMBDAS:
        calibration_jobs.append(_job("cmlp", data, config, config["calibration_seed"], device, workers, lambda_value=lam))
    for lam in LORENZ_GVAR_LAMBDAS:
        for gamma in LORENZ_GVAR_GAMMAS:
            calibration_jobs.append(
                _job("gvar", data, config, config["calibration_seed"], device, workers, lambda_value=lam, gamma=gamma, stability=False)
            )
    calibration = run_cached_jobs(
        calibration_jobs,
        cache_dir=cache / "calibration",
        force=force,
        workers=workers,
    )
    selected_x = choose_lambda([row for row in calibration if row["method"] == "Xneural VAR"], truth)
    selected_c = choose_lambda([row for row in calibration if row["method"] == "NGC (cMLP)"], truth)
    selected_g = choose_gvar_params([row for row in calibration if row["method"] == "GVAR"], truth)

    evaluation_jobs = []
    for seed in config["evaluation_seeds"]:
        for lam in LORENZ_XNEURAL_LAMBDAS:
            evaluation_jobs.append(_job("xneural", data, config, seed, device, workers, lambda_value=lam))
        for lam in LORENZ_CMLP_LAMBDAS:
            evaluation_jobs.append(_job("cmlp", data, config, seed, device, workers, lambda_value=lam))
        evaluation_jobs.append(
            _job("gvar", data, config, seed, device, workers, lambda_value=selected_g[0], gamma=selected_g[1], stability=True)
        )
    evaluation = run_cached_jobs(
        evaluation_jobs,
        cache_dir=cache / "evaluation",
        force=force,
        workers=workers,
    )

    x_rows = [row for row in evaluation if row["method"] == "Xneural VAR"]
    c_rows = [row for row in evaluation if row["method"] == "NGC (cMLP)"]
    g_rows = [row for row in evaluation if row["method"] == "GVAR"]
    linear_runs, linear_details = _linear_runs(
        {config["evaluation_seeds"][0]: data},
        truth,
        order=config["order"],
        fdr_level=0.05,
        true_coeffs=None,
    )

    methods = [
        aggregate_method_runs("Xneural VAR", _path_runs(x_rows, truth, config["evaluation_seeds"], selected_x)),
        aggregate_method_runs("NGC (cMLP)", _path_runs(c_rows, truth, config["evaluation_seeds"], selected_c)),
        aggregate_method_runs("GVAR", _gvar_runs(g_rows, truth)),
        aggregate_method_runs("linear VAR (F-test, BH)", linear_runs),
    ]
    return {
        "name": f"Lorenz-96 F={forcing:g}",
        "forcing": forcing,
        "config": config,
        "ground_truth": truth.tolist(),
        "true_edges_all": int(truth.sum()),
        "true_edges_offdiag": int(truth[~np.eye(config["p"], dtype=bool)].sum()),
        "selected_hyperparameters": {
            "xneural_lambda": selected_x,
            "cmlp_lambda": selected_c,
            "gvar_lambda": selected_g[0],
            "gvar_gamma": selected_g[1],
            "linear_var_fdr_q": 0.05,
        },
        "methods": methods,
        "xneural_lambda_path": lambda_path_summary(x_rows, truth, config["evaluation_seeds"]),
        "cmlp_lambda_path": lambda_path_summary(c_rows, truth, config["evaluation_seeds"]),
        "representative_linear_pvalues": linear_details[config["evaluation_seeds"][0]]["pvalues"].tolist(),
    }


def run_linear_condition(*, force: bool, workers: int, device: str) -> dict[str, Any]:
    config = dict(LINEAR_CONFIG)
    config["order"] = config["fit_order"]
    coeffs = true_linear_var_coefficients(config["p"], config["true_order"])
    truth = (np.abs(coeffs).sum(axis=0) > 0).astype(np.int64)

    def make_data(seed: int) -> np.ndarray:
        return standardize_full(
            simulate_linear_var(
                coeffs=coeffs,
                T=config["T"],
                burn_in=config["burn_in"],
                noise_sd=config["noise_sd"],
                innovation_correlation=config["innovation_correlation"],
                seed=seed,
            )
        )

    calibration_data = make_data(config["calibration_seed"])
    cache = RAW_DIR / "linear_var_lag1"
    calibration_jobs = []
    for lam in LINEAR_XNEURAL_LAMBDAS:
        calibration_jobs.append(
            _job("xneural", calibration_data, config, config["calibration_seed"], device, workers, lambda_value=lam, true_coeffs=coeffs)
        )
    for lam in LINEAR_CMLP_LAMBDAS:
        calibration_jobs.append(_job("cmlp", calibration_data, config, config["calibration_seed"], device, workers, lambda_value=lam))
    for lam in LINEAR_GVAR_LAMBDAS:
        for gamma in LINEAR_GVAR_GAMMAS:
            calibration_jobs.append(
                _job(
                    "gvar",
                    calibration_data,
                    config,
                    config["calibration_seed"],
                    device,
                    workers,
                    lambda_value=lam,
                    gamma=gamma,
                    stability=False,
                    true_coeffs=coeffs,
                )
            )
    calibration = run_cached_jobs(calibration_jobs, cache_dir=cache / "calibration", force=force, workers=workers)
    selected_x = choose_lambda([row for row in calibration if row["method"] == "Xneural VAR"], truth)
    selected_c = choose_lambda([row for row in calibration if row["method"] == "NGC (cMLP)"], truth)
    selected_g = choose_gvar_params([row for row in calibration if row["method"] == "GVAR"], truth)

    datasets = {seed: make_data(seed) for seed in config["evaluation_seeds"]}
    evaluation_jobs = []
    for seed, data in datasets.items():
        for lam in LINEAR_XNEURAL_LAMBDAS:
            evaluation_jobs.append(_job("xneural", data, config, seed, device, workers, lambda_value=lam, true_coeffs=coeffs))
        for lam in LINEAR_CMLP_LAMBDAS:
            evaluation_jobs.append(_job("cmlp", data, config, seed, device, workers, lambda_value=lam))
        evaluation_jobs.append(
            _job(
                "gvar",
                data,
                config,
                seed,
                device,
                workers,
                lambda_value=selected_g[0],
                gamma=selected_g[1],
                stability=True,
                true_coeffs=coeffs,
            )
        )
    evaluation = run_cached_jobs(evaluation_jobs, cache_dir=cache / "evaluation", force=force, workers=workers)

    x_rows = [row for row in evaluation if row["method"] == "Xneural VAR"]
    c_rows = [row for row in evaluation if row["method"] == "NGC (cMLP)"]
    g_rows = [row for row in evaluation if row["method"] == "GVAR"]
    linear_runs, linear_details = _linear_runs(
        datasets,
        truth,
        order=config["fit_order"],
        fdr_level=config["fdr_level"],
        true_coeffs=coeffs,
    )

    methods = [
        aggregate_method_runs("Xneural VAR", _path_runs(x_rows, truth, config["evaluation_seeds"], selected_x)),
        aggregate_method_runs("NGC (cMLP)", _path_runs(c_rows, truth, config["evaluation_seeds"], selected_c)),
        aggregate_method_runs("GVAR", _gvar_runs(g_rows, truth)),
        aggregate_method_runs("linear VAR (F-test, BH)", linear_runs),
    ]
    return {
        "name": f"Linear VAR({config['true_order']})",
        "config": config,
        "ground_truth": truth.tolist(),
        "true_coefficients": coeffs.tolist(),
        "spectral_radius": _companion_radius(coeffs),
        "true_edges_all": int(truth.sum()),
        "true_edges_offdiag": int(truth[~np.eye(config["p"], dtype=bool)].sum()),
        "selected_hyperparameters": {
            "xneural_lambda": selected_x,
            "cmlp_lambda": selected_c,
            "gvar_lambda": selected_g[0],
            "gvar_gamma": selected_g[1],
            "linear_var_fdr_q": config["fdr_level"],
        },
        "methods": methods,
        "xneural_lambda_path": lambda_path_summary(x_rows, truth, config["evaluation_seeds"]),
        "cmlp_lambda_path": lambda_path_summary(c_rows, truth, config["evaluation_seeds"]),
        "representative_linear_pvalues": linear_details[config["evaluation_seeds"][0]]["pvalues"].tolist(),
    }


def _method_lookup(condition: dict[str, Any], method: str) -> dict[str, Any]:
    return next(item for item in condition["methods"] if item["method"] == method)


def write_csvs(payload: dict[str, Any]) -> None:
    method_rows = []
    lambda_rows = []
    for condition in payload["conditions"]:
        for method in condition["methods"]:
            row = {"condition": condition["name"], "method": method["method"]}
            for metric, stats in method["summary"].items():
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_ci95"] = stats["ci95"]
            method_rows.append(row)
        for entry in condition["xneural_lambda_path"]:
            lambda_rows.append(
                {
                    "condition": condition["name"],
                    "lambda": entry["lambda"],
                    "selected_edges_mean": entry["selected_edges"]["mean"],
                    "precision_mean": entry["precision"]["mean"],
                    "recall_mean": entry["recall"]["mean"],
                    "f1_mean": entry["f1"]["mean"],
                    "balanced_accuracy_mean": entry["balanced_accuracy"]["mean"],
                }
            )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        columns = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(METHOD_CSV, method_rows)
    write_csv(LAMBDA_CSV, lambda_rows)


def plot_aggregate_results(payload: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = ["Xneural VAR", "NGC (cMLP)", "GVAR", "linear VAR (F-test, BH)"]
    colors = ["#16697A", "#489FB5", "#E09F3E", "#9E2A2B"]
    for condition in payload["conditions"]:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
        for ax, metric, label in zip(axes, ["auprc", "f1", "selected_edges"], ["AUPRC", "F1", "Selected edges"]):
            means = [_method_lookup(condition, method)["summary"][metric]["mean"] for method in methods]
            errors = [_method_lookup(condition, method)["summary"][metric]["ci95"] for method in methods]
            x = np.arange(len(methods))
            ax.bar(x, means, yerr=errors, color=colors, capsize=3, edgecolor="#222222", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(["Xneural", "NGC", "GVAR", "linear VAR"], rotation=20, ha="right")
            ax.set_title(label, fontweight="semibold")
            ax.grid(axis="y", alpha=0.25)
            if metric != "selected_edges":
                ax.set_ylim(0.0, 1.05)
        fig.suptitle(condition["name"], fontsize=14, fontweight="semibold")
        fig.savefig(FIG_DIR / f"{condition['name'].lower().replace(' ', '_').replace('=', '')}_comparison.png", dpi=300)
        plt.close(fig)

        path = condition["xneural_lambda_path"]
        fig, axis_left = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        lambdas = [entry["lambda"] for entry in path]
        f1 = [entry["f1"]["mean"] for entry in path]
        edges = [entry["selected_edges"]["mean"] for entry in path]
        axis_left.plot(lambdas, f1, marker="o", color="#16697A", linewidth=2, label="F1")
        axis_left.set_xscale("log")
        axis_left.set_xlabel(r"regularization strength $\lambda$")
        axis_left.set_ylabel("F1", color="#16697A")
        axis_left.set_ylim(0.0, 1.05)
        axis_right = axis_left.twinx()
        axis_right.plot(lambdas, edges, marker="s", color="#9E2A2B", linewidth=2, label="selected edges")
        axis_right.set_ylabel("selected off-diagonal edges", color="#9E2A2B")
        axis_left.grid(alpha=0.25)
        axis_left.set_title(f"Xneural VAR exact-zero path: {condition['name']}", fontweight="semibold")
        fig.savefig(FIG_DIR / f"{condition['name'].lower().replace(' ', '_').replace('=', '')}_xneural_path.png", dpi=300)
        plt.close(fig)


def make_representative_visualizations(payload: dict[str, Any], device: str) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))

    for condition in [item for item in payload["conditions"] if item["name"].startswith("Lorenz")]:
        forcing = float(condition["forcing"])
        config = dict(condition["config"])
        data = standardize_full(
            simulate_lorenz96_ngc(
                p=config["p"],
                T=config["T"],
                forcing=forcing,
                delta_t=config["delta_t"],
                observation_noise_sd=config["observation_noise_sd"],
                burn_in=config["burn_in"],
                seed=config["data_seed"],
            )
        )
        _, result = _fit_xneural(
            data,
            config,
            condition["selected_hyperparameters"]["xneural_lambda"],
            config["evaluation_seeds"][0],
            device,
        )
        figure, _ = plot_causal_gate_by_lag(
            result,
            summary="norm",
            percentile=99,
            tick_label_step=2,
            title=f"Xneural VAR causal gate: Lorenz-96 F={forcing:g}",
            save_path=FIG_DIR / f"lorenz96_f{forcing:g}_gate_by_lag_benchmark.png",
            show=False,
        )
        plt.close(figure)

    linear = next(item for item in payload["conditions"] if item["name"].startswith("Linear"))
    config = dict(linear["config"])
    true_coeffs = np.asarray(linear["true_coefficients"], dtype=np.float64)
    data = standardize_full(
        simulate_linear_var(
            coeffs=true_coeffs,
            T=config["T"],
            burn_in=config["burn_in"],
            noise_sd=config["noise_sd"],
            innovation_correlation=config["innovation_correlation"],
            seed=config["evaluation_seeds"][0],
        )
    )
    _, result = _fit_xneural(
        data,
        config,
        linear["selected_hyperparameters"]["xneural_lambda"],
        config["evaluation_seeds"][0],
        device,
    )
    figure, _ = plot_causal_gate_by_lag(
        result,
        summary="norm",
        percentile=99,
        title=f"Xneural VAR causal gate: linear VAR({config['true_order']})",
        save_path=FIG_DIR / "linear_var_gate_by_lag_benchmark.png",
        show=False,
    )
    plt.close(figure)
    true_edges = np.argwhere((np.abs(true_coeffs).sum(axis=0) > 0) & ~np.eye(config["p"], dtype=bool))
    chosen_edges = [tuple(map(int, edge)) for edge in true_edges[:6]]
    figure, _ = plot_edge_lag_boxplots(
        result,
        edges=chosen_edges,
        value="signed",
        title="Signed effective coefficients on true edges",
        save_path=FIG_DIR / "linear_var_signed_boxplots_benchmark.png",
        show=False,
    )
    plt.close(figure)


def _format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_method_table(condition: dict[str, Any]) -> str:
    header = ["method", "AUROC", "AUPRC", "F1", "BA", "edges", "sign"]
    rows = []
    for method in condition["methods"]:
        summary = method["summary"]
        rows.append(
            [
                method["method"],
                _format_number(summary["auroc"]["mean"]),
                _format_number(summary["auprc"]["mean"]),
                _format_number(summary["f1"]["mean"]),
                _format_number(summary["balanced_accuracy"]["mean"]),
                _format_number(summary["selected_edges"]["mean"], 1),
                _format_number(summary["sign_accuracy"]["mean"]) if "sign_accuracy" in summary else "--",
            ]
        )
    widths = [max(len(header[col]), *(len(row[col]) for row in rows)) for col in range(len(header))]
    lines = ["  ".join(header[col].ljust(widths[col]) for col in range(len(header)))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(row[col].ljust(widths[col]) for col in range(len(header))) for row in rows)
    return "\n".join(lines)


def _notebook_cell(cell_type: str, source: str, *, execution_count: int | None = None, output: str | None = None) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = [] if output is None else [
            {"name": "stdout", "output_type": "stream", "text": output.splitlines(keepends=True)}
        ]
    return cell


def make_notebook(payload: dict[str, Any]) -> None:
    config_output = json.dumps(payload["protocol"], ensure_ascii=False, indent=2)
    cells = [
        _notebook_cell(
            "markdown",
            "# Xneural VAR: fair Granger-causality benchmarks\n\n"
            "This notebook is generated from the completed experiment and contains only the reproducible protocol, "
            "result loading, final tables, and figures. No failed trial cells are included.\n",
        ),
        _notebook_cell(
            "markdown",
            "## Decision rules\n\n"
            "- Xneural VAR and NGC/cMLP: an edge is present iff its proximal group is exactly nonzero. "
            "AUROC/AUPRC use the regularization path, not a post-hoc weight threshold.\n"
            "- GVAR: continuous coefficient strength for AUROC/AUPRC and the original time-reversal stability rule for a binary graph.\n"
            "- Linear VAR: joint F-test over all fitted lags and Benjamini-Hochberg FDR control at q=0.05.\n"
            "- Unified comparison excludes diagonal self-edges; all-edge Lorenz metrics are retained in the JSON for Neural-GC comparability.\n",
        ),
        _notebook_cell("code", "PROTOCOL = " + repr(payload["protocol"]) + "\nPROTOCOL\n", execution_count=1, output=config_output + "\n"),
        _notebook_cell(
            "code",
            "from pathlib import Path\nimport json\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'experiments').exists():\n    ROOT = ROOT.parent\n"
            "RESULT_PATH = ROOT / 'experiments' / 'benchmark_results.json'\n"
            "RESULTS = json.loads(RESULT_PATH.read_text(encoding='utf-8'))\n"
            "print('loaded:', RESULT_PATH)\n",
            execution_count=2,
            output="loaded: experiments/benchmark_results.json\n",
        ),
    ]
    execution = 3
    for condition in payload["conditions"]:
        table = format_method_table(condition)
        cells.append(_notebook_cell("markdown", f"## {condition['name']}\n"))
        cells.append(
            _notebook_cell(
                "code",
                f"condition = next(c for c in RESULTS['conditions'] if c['name'] == {condition['name']!r})\n"
                "print(condition['selected_hyperparameters'])\n",
                execution_count=execution,
                output=str(condition["selected_hyperparameters"]) + "\n",
            )
        )
        execution += 1
        cells.append(
            _notebook_cell(
                "code",
                "# Mean metrics across the five predeclared evaluation seeds.\n"
                "from run_all_benchmarks import format_method_table\n"
                "print(format_method_table(condition))\n",
                execution_count=execution,
                output=table + "\n",
            )
        )
        execution += 1
        image_stem = condition["name"].lower().replace(" ", "_").replace("=", "")
        cells.append(_notebook_cell("markdown", f"![comparison](../figures/{image_stem}_comparison.png)\n\n![path](../figures/{image_stem}_xneural_path.png)\n"))

    cells.extend(
        [
            _notebook_cell("markdown", "## Lag-wise gates and sign distributions\n\n![F10 gate](../figures/lorenz96_f10_gate_by_lag_benchmark.png)\n\n![F40 gate](../figures/lorenz96_f40_gate_by_lag_benchmark.png)\n\n![linear gate](../figures/linear_var_gate_by_lag_benchmark.png)\n\n![signed coefficients](../figures/linear_var_signed_boxplots_benchmark.png)\n"),
            _notebook_cell(
                "markdown",
                "## Full rerun\n\nRun the command below from the repository root to recompute every fit. "
                "Cached fits are used when `--force` is omitted.\n",
            ),
            _notebook_cell(
                "code",
                "# %run experiments/run_all_benchmarks.py --force --workers 3\n",
                execution_count=None,
            ),
        ]
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(NOTEBOOK_PATH, notebook)


def run_all_benchmarks(*, force: bool = False, workers: int = 1, device: str | None = None) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    started = time.perf_counter()
    conditions = [
        run_linear_condition(force=force, workers=workers, device=device),
        run_lorenz_condition(10.0, force=force, workers=workers, device=device),
        run_lorenz_condition(40.0, force=force, workers=workers, device=device),
    ]
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "protocol": {
            "primary_claims": [
                "signed effective-coefficient interpretability",
                "post-hoc-threshold-free exact sparsity for Xneural VAR and NGC/cMLP",
                "Granger graph recovery on linear VAR and Neural-GC Lorenz-96 settings",
            ],
            "lorenz_reference": "Tank et al. Neural Granger Causality, published TPAMI version; T=500 condition",
            "lorenz_config": LORENZ_CONFIG,
            "linear_config": LINEAR_CONFIG,
            "unified_evaluation": "off-diagonal target-source pairs",
            "xneural_ngc_binary_rule": "group norm > 0 exactly; no post-hoc threshold and no top-k",
            "linear_var_binary_rule": "joint all-lag F-test with Benjamini-Hochberg FDR q=0.05",
            "gvar_binary_rule": "time-reversal stability-based quantile selection, Q=20",
            "calibration": "one separate calibration seed; five evaluation seeds are not used for hyperparameter selection",
        },
        "device": device,
        "conditions": conditions,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    write_json(RESULT_JSON, payload)
    write_csvs(payload)
    plot_aggregate_results(payload)
    make_representative_visualizations(payload, device)
    make_notebook(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the threshold-free Xneural VAR benchmark suite.")
    parser.add_argument("--force", action="store_true", help="recompute fits instead of using the versioned cache")
    parser.add_argument("--workers", type=int, default=1, help="number of independent CPU fit workers")
    parser.add_argument("--device", default=None, help="torch device, for example cpu or cuda")
    args = parser.parse_args()
    payload = run_all_benchmarks(force=args.force, workers=max(args.workers, 1), device=args.device)
    for condition in payload["conditions"]:
        print("\n" + condition["name"])
        print(format_method_table(condition))
    print(f"\nWrote {RESULT_JSON}")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
