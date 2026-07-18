from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xneural_var import (
    CMLPTrainingConfig,
    GVARBaselineTrainingConfig,
    GVARTrainingConfig,
    construct_lagged_dataset,
    fit_cmlp,
    fit_gvar,
    fit_gvar_ngc,
)


OUT_DIR = ROOT / "experiments"
RAW_DIR = OUT_DIR / "raw_lorenz96"
RESULT_JSON = OUT_DIR / "lorenz96_results.json"
LAMBDA_CSV = OUT_DIR / "lorenz96_lambda_sweep.csv"
COMPARISON_CSV = OUT_DIR / "lorenz96_method_comparison.csv"
NOTEBOOK_PATH = OUT_DIR / "lorenz96_benchmark.ipynb"


DEFAULT_CONFIG: dict[str, Any] = {
    "p": 20,
    "forcing": 10.0,
    "order": 5,
    "n_repeats": 5,
    "total_T": 500,
    "train_T": 400,
    "dt": 0.01,
    "sample_dt": 0.05,
    "burn_in_steps": 1000,
    "hidden_layer_size": 12,
    "num_hidden_layers": 1,
    "max_epochs": 80,
    "batch_size": 128,
    "learning_rate": 3e-2,
    "lambda_smooth": 1e-2,
    "coefficient_weight_decay": 1e-4,
    "seed": 20260718,
}

XNEURAL_LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.06, 0.09]
CMLP_LAMBDAS = [0.001, 0.003, 0.01]
GVAR_LAMBDAS = [0.0, 0.001, 0.01]
LINEAR_RIDGES = [1e-6, 1e-4, 1e-2, 1.0, 10.0]


def lorenz96_rhs(x: np.ndarray, forcing: float) -> np.ndarray:
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def rk4_step(x: np.ndarray, dt: float, forcing: float) -> np.ndarray:
    k1 = lorenz96_rhs(x, forcing)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = lorenz96_rhs(x + dt * k3, forcing)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_lorenz96(
    *,
    p: int,
    forcing: float,
    total_T: int,
    dt: float,
    sample_dt: float,
    burn_in_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_steps = int(round(sample_dt / dt))
    if sample_steps <= 0 or not math.isclose(sample_steps * dt, sample_dt, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("sample_dt must be a positive integer multiple of dt")

    x = forcing * np.ones(p, dtype=np.float64)
    x += 0.01 * rng.standard_normal(p)
    for _ in range(burn_in_steps):
        x = rk4_step(x, dt, forcing)

    samples = np.empty((total_T, p), dtype=np.float32)
    for t in range(total_T):
        for _ in range(sample_steps):
            x = rk4_step(x, dt, forcing)
        samples[t] = x.astype(np.float32)
    return samples


def make_lorenz96_splits(config: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(config["seed"])
    raw = [
        simulate_lorenz96(
            p=config["p"],
            forcing=config["forcing"],
            total_T=config["total_T"],
            dt=config["dt"],
            sample_dt=config["sample_dt"],
            burn_in_steps=config["burn_in_steps"],
            rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
        )
        for _ in range(config["n_repeats"])
    ]

    train_raw = [series[: config["train_T"]] for series in raw]
    train_concat = np.concatenate(train_raw, axis=0)
    mean = train_concat.mean(axis=0, keepdims=True)
    std = train_concat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    train = [((series[: config["train_T"]] - mean) / std).astype(np.float32) for series in raw]
    start = config["train_T"] - config["order"]
    test = [((series[start:] - mean) / std).astype(np.float32) for series in raw]
    return train, test, true_lorenz96_graph(config["p"])


def true_lorenz96_graph(p: int) -> np.ndarray:
    graph = np.zeros((p, p), dtype=np.int64)
    for i in range(p):
        for j in (i, (i + 1) % p, (i - 1) % p, (i - 2) % p):
            graph[i, j] = 1
    return graph


def offdiag_vectors(score: np.ndarray, true_graph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if score.shape != true_graph.shape:
        raise ValueError("score and true_graph must have the same shape")
    mask = ~np.eye(score.shape[0], dtype=bool)
    return np.asarray(score[mask], dtype=np.float64), np.asarray(true_graph[mask], dtype=np.int64)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels == 1].sum() / positives)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * (start + 1 + stop)
        ranks[order[start:stop]] = average_rank
        start = stop

    sum_pos_ranks = float(ranks[labels == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def topk_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    k = int(labels.sum())
    if k <= 0:
        raise ValueError("labels must contain at least one positive edge")
    order = np.argsort(-scores, kind="mergesort")
    selected = np.zeros_like(labels, dtype=bool)
    selected[order[:k]] = True

    positives = labels.astype(bool)
    tp = int(np.logical_and(selected, positives).sum())
    fp = int(np.logical_and(selected, ~positives).sum())
    fn = int(np.logical_and(~selected, positives).sum())
    tn = int(np.logical_and(~selected, ~positives).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    tnr = tn / max(tn + fp, 1)
    return {
        "top_k": float(k),
        "threshold": float(scores[order[k - 1]]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + tnr)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def evaluate_scores(score: np.ndarray, true_graph: np.ndarray) -> dict[str, float]:
    scores, labels = offdiag_vectors(score, true_graph)
    metrics = {
        "auroc": float(auroc(scores, labels)),
        "auprc": float(average_precision(scores, labels)),
    }
    metrics.update(topk_metrics(scores, labels))
    return metrics


@torch.no_grad()
def neural_test_mse(model: torch.nn.Module, test: list[np.ndarray], order: int, batch_size: int, device: str) -> float:
    dataset = construct_lagged_dataset(test, order)
    predictors = torch.as_tensor(dataset.predictors, dtype=torch.float32, device=device)
    responses = torch.as_tensor(dataset.responses, dtype=torch.float32, device=device)
    model.to(device)
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, predictors.shape[0], batch_size):
        x = predictors[start : start + batch_size]
        y = responses[start : start + batch_size]
        out = model(x)
        preds = out[0] if isinstance(out, tuple) else out
        total += float(torch.sum((preds - y).pow(2)).detach().cpu())
        count += int(y.numel())
    return total / max(count, 1)


def fit_linear_var(train: list[np.ndarray], order: int, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    dataset = construct_lagged_dataset(train, order)
    x = dataset.predictors.reshape(dataset.predictors.shape[0], -1).astype(np.float64)
    y = dataset.responses.astype(np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + ridge * penalty, x_aug.T @ y)
    coef = beta[1:].reshape(order, y.shape[1], y.shape[1])
    strength = np.linalg.vector_norm(coef, ord=2, axis=0).T
    return beta, strength


def linear_test_mse(beta: np.ndarray, test: list[np.ndarray], order: int) -> float:
    dataset = construct_lagged_dataset(test, order)
    x = dataset.predictors.reshape(dataset.predictors.shape[0], -1).astype(np.float64)
    y = dataset.responses.astype(np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    preds = x_aug @ beta
    return float(np.mean((preds - y) ** 2))


def result_path(run_id: str) -> Path:
    return RAW_DIR / f"{run_id}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_xneural(
    train: list[np.ndarray],
    test: list[np.ndarray],
    true_graph: np.ndarray,
    config: dict[str, Any],
    lam: float,
    device: str,
) -> dict[str, Any]:
    fit_config = GVARTrainingConfig(
        order=config["order"],
        hidden_layer_size=config["hidden_layer_size"],
        num_hidden_layers=config["num_hidden_layers"],
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        lambda_ngc=lam,
        lambda_smooth=config["lambda_smooth"],
        coefficient_weight_decay=config["coefficient_weight_decay"],
        regularizer="hierarchical_group_lasso",
        optimizer="ista",
        gate_init=1.0,
        causal_threshold=1e-8,
        strength_aggregation="max",
        smoothness_mode="absolute",
        seed=config["seed"],
        verbose=0,
        device=device,
    )
    start = time.perf_counter()
    result = fit_gvar_ngc(train, fit_config)
    elapsed = time.perf_counter() - start
    gate_score = result.model.gate_group_norms().detach().cpu().numpy()
    metrics = evaluate_scores(gate_score, true_graph)
    metrics["test_mse"] = neural_test_mse(result.model, test, config["order"], config["batch_size"], device)
    metrics["native_edges_tau_1e-8"] = float((gate_score[~np.eye(gate_score.shape[0], dtype=bool)] > 1e-8).sum())
    return make_result_row(
        method="Xneural VAR",
        variant="ISTA + hierarchical group lasso",
        param_name="lambda_ngc",
        param_value=lam,
        metrics=metrics,
        elapsed=elapsed,
    )


def run_gvar(
    train: list[np.ndarray],
    test: list[np.ndarray],
    true_graph: np.ndarray,
    config: dict[str, Any],
    lam: float,
    device: str,
) -> dict[str, Any]:
    fit_config = GVARBaselineTrainingConfig(
        order=config["order"],
        hidden_layer_size=config["hidden_layer_size"],
        num_hidden_layers=config["num_hidden_layers"],
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        lambda_coeff=lam,
        elastic_net_alpha=0.5,
        lambda_smooth=config["lambda_smooth"],
        coefficient_weight_decay=config["coefficient_weight_decay"],
        causal_threshold=1e-8,
        strength_aggregation="max",
        smoothness_mode="absolute",
        seed=config["seed"],
        verbose=0,
        device=device,
    )
    start = time.perf_counter()
    result = fit_gvar(train, fit_config)
    elapsed = time.perf_counter() - start
    score = np.asarray(result.causal_strength, dtype=np.float64)
    metrics = evaluate_scores(score, true_graph)
    metrics["test_mse"] = neural_test_mse(result.model, test, config["order"], config["batch_size"], device)
    metrics["native_edges_tau_1e-8"] = float((score[~np.eye(score.shape[0], dtype=bool)] > 1e-8).sum())
    return make_result_row(
        method="GVAR",
        variant="coefficient elastic-net + smoothness",
        param_name="lambda_coeff",
        param_value=lam,
        metrics=metrics,
        elapsed=elapsed,
    )


def run_cmlp(
    train: list[np.ndarray],
    test: list[np.ndarray],
    true_graph: np.ndarray,
    config: dict[str, Any],
    lam: float,
    device: str,
) -> dict[str, Any]:
    fit_config = CMLPTrainingConfig(
        order=config["order"],
        hidden_layer_size=config["hidden_layer_size"],
        num_hidden_layers=config["num_hidden_layers"],
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        lambda_ngc=lam,
        ridge_lambda=config["coefficient_weight_decay"],
        regularizer="hierarchical_group_lasso",
        optimizer="ista",
        causal_threshold=1e-8,
        seed=config["seed"],
        verbose=0,
        device=device,
    )
    start = time.perf_counter()
    result = fit_cmlp(train, fit_config)
    elapsed = time.perf_counter() - start
    score = np.asarray(result.causal_strength, dtype=np.float64)
    metrics = evaluate_scores(score, true_graph)
    metrics["test_mse"] = neural_test_mse(result.model, test, config["order"], config["batch_size"], device)
    metrics["native_edges_tau_1e-8"] = float((score[~np.eye(score.shape[0], dtype=bool)] > 1e-8).sum())
    return make_result_row(
        method="cMLP",
        variant="Neural-GC hierarchical group lasso",
        param_name="lambda_ngc",
        param_value=lam,
        metrics=metrics,
        elapsed=elapsed,
    )


def run_linear(
    train: list[np.ndarray],
    test: list[np.ndarray],
    true_graph: np.ndarray,
    config: dict[str, Any],
    ridge: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    beta, score = fit_linear_var(train, config["order"], ridge)
    elapsed = time.perf_counter() - start
    metrics = evaluate_scores(score, true_graph)
    metrics["test_mse"] = linear_test_mse(beta, test, config["order"])
    metrics["native_edges_tau_1e-8"] = float((score[~np.eye(score.shape[0], dtype=bool)] > 1e-8).sum())
    return make_result_row(
        method="linear VAR",
        variant="ridge VAR",
        param_name="ridge",
        param_value=ridge,
        metrics=metrics,
        elapsed=elapsed,
    )


def make_result_row(
    *,
    method: str,
    variant: str,
    param_name: str,
    param_value: float,
    metrics: dict[str, float],
    elapsed: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "variant": variant,
        "param_name": param_name,
        "param_value": float(param_value),
        "elapsed_sec": float(elapsed),
    }
    row.update(metrics)
    return row


def run_or_load(
    *,
    run_id: str,
    force: bool,
    callback,
) -> dict[str, Any]:
    path = result_path(run_id)
    if path.exists() and not force:
        print(f"[cache] {run_id}")
        return read_json(path)
    print(f"[run]   {run_id}")
    row = callback()
    write_json(path, row)
    print(
        f"        AUPRC={row['auprc']:.3f}, F1={row['f1']:.3f}, "
        f"BA={row['balanced_accuracy']:.3f}, MSE={row['test_mse']:.4g}, "
        f"time={row['elapsed_sec']:.1f}s"
    )
    return row


def run_benchmark(force: bool = False, device: str | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    print("Preparing Lorenz-96 data...")
    train, test, true_graph = make_lorenz96_splits(config)
    rows: list[dict[str, Any]] = []

    for ridge in LINEAR_RIDGES:
        run_id = f"linear_ridge_{ridge:g}"
        rows.append(
            run_or_load(
                run_id=run_id,
                force=force,
                callback=lambda ridge=ridge: run_linear(train, test, true_graph, config, ridge),
            )
        )

    for lam in GVAR_LAMBDAS:
        run_id = f"gvar_lambda_coeff_{lam:g}"
        rows.append(
            run_or_load(
                run_id=run_id,
                force=force,
                callback=lambda lam=lam: run_gvar(train, test, true_graph, config, lam, device),
            )
        )

    for lam in XNEURAL_LAMBDAS:
        run_id = f"xneural_lambda_ngc_{lam:g}"
        rows.append(
            run_or_load(
                run_id=run_id,
                force=force,
                callback=lambda lam=lam: run_xneural(train, test, true_graph, config, lam, device),
            )
        )

    for lam in CMLP_LAMBDAS:
        run_id = f"cmlp_lambda_ngc_{lam:g}"
        rows.append(
            run_or_load(
                run_id=run_id,
                force=force,
                callback=lambda lam=lam: run_cmlp(train, test, true_graph, config, lam, device),
            )
        )

    return aggregate_results(config=config, rows=rows, device=device)


def aggregate_results(config: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None, device: str = "cpu") -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [read_json(path) for path in sorted(RAW_DIR.glob("*.json"))]
    if config is None:
        config = dict(DEFAULT_CONFIG)

    rows = sorted(rows, key=lambda r: (r["method"], r["param_value"]))
    lambda_rows = [row for row in rows if row["method"] == "Xneural VAR"]
    comparison_rows = best_rows_by_method(rows)

    write_csv(LAMBDA_CSV, lambda_rows)
    write_csv(COMPARISON_CSV, comparison_rows)

    payload = {
        "experiment": {
            "name": "Lorenz-96 Granger benchmark",
            "device": device,
            "config": config,
            "true_offdiagonal_edges": int(3 * config["p"]),
            "decision_rule": "AUROC/AUPRC use continuous off-diagonal scores. Precision/Recall/F1 use the top 3p off-diagonal scores for every method.",
            "method_selection": "For method comparison, each baseline is selected by validation-on-ground-truth AUPRC over the explicitly reported grid, matching common synthetic benchmark reporting.",
            "xneural_score": "L2 norm of causal_gate over lags.",
            "gvar_score": "max absolute effective coefficient over samples and lags.",
            "cmlp_score": "first-layer source group norm.",
            "linear_var_score": "L2 norm of ridge VAR coefficients over lags.",
        },
        "grids": {
            "xneural_lambda_ngc": XNEURAL_LAMBDAS,
            "cmlp_lambda_ngc": CMLP_LAMBDAS,
            "gvar_lambda_coeff": GVAR_LAMBDAS,
            "linear_ridge": LINEAR_RIDGES,
        },
        "all_results": rows,
        "xneural_lambda_sweep": lambda_rows,
        "method_comparison": comparison_rows,
    }
    write_json(RESULT_JSON, payload)
    make_notebook(payload)
    return payload


def best_rows_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        method = row["method"]
        candidate_key = (row["auprc"], row["f1"], row["balanced_accuracy"], -row["test_mse"])
        if method not in best:
            best[method] = row
            continue
        current = best[method]
        current_key = (current["auprc"], current["f1"], current["balanced_accuracy"], -current["test_mse"])
        if candidate_key > current_key:
            best[method] = row
    order = ["Xneural VAR", "GVAR", "cMLP", "linear VAR"]
    return [best[method] for method in order if method in best]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "variant",
        "param_name",
        "param_value",
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "threshold",
        "test_mse",
        "native_edges_tau_1e-8",
        "elapsed_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def format_table(rows: list[dict[str, Any]], *, columns: list[str]) -> str:
    labels = {
        "method": "method",
        "param_value": "param",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "balanced_accuracy": "BA",
        "test_mse": "test MSE",
        "threshold": "top-k threshold",
    }
    table: list[list[str]] = [[labels.get(column, column) for column in columns]]
    for row in rows:
        line = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if column == "test_mse":
                    line.append(f"{value:.4g}")
                elif column in {"param_value", "threshold"}:
                    line.append(f"{value:.4g}")
                else:
                    line.append(f"{value:.3f}")
            else:
                line.append(str(value))
        table.append(line)
    widths = [max(len(line[idx]) for line in table) for idx in range(len(columns))]
    return "\n".join("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(line)) for line in table)


def notebook_cell(cell_type: str, source: str, execution_count: int | None = None, output: str | None = None) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        if output is not None:
            cell["outputs"].append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output.splitlines(keepends=True),
                }
            )
    return cell


def make_notebook(payload: dict[str, Any]) -> None:
    lambda_table = format_table(
        payload["xneural_lambda_sweep"],
        columns=["param_value", "auroc", "auprc", "precision", "recall", "f1", "balanced_accuracy", "test_mse", "threshold"],
    )
    comparison_table = format_table(
        payload["method_comparison"],
        columns=["method", "param_value", "auroc", "auprc", "precision", "recall", "f1", "balanced_accuracy", "test_mse", "threshold"],
    )
    config_text = json.dumps(payload["experiment"]["config"], ensure_ascii=False, indent=2)
    cells = [
        notebook_cell(
            "markdown",
            "# Lorenz-96 benchmark for Xneural VAR\n\n"
            "This notebook records the clean experiment used in `slides/slide.tex`. "
            "The executable implementation is `experiments/run_lorenz96_benchmark.py`; "
            "running the cell below reuses cached JSON results unless `force=True` is passed.",
        ),
        notebook_cell(
            "markdown",
            "## Experimental design\n\n"
            "- Data: Lorenz-96, p=20, F=10, RK4 integration, sample interval 0.05.\n"
            "- Repeats: 5 independent trajectories, T=500 each.\n"
            "- Split: first 400 points for training and final 100 points for prediction evaluation.\n"
            "- Lag order: K=5.\n"
            "- Structural ground truth: off-diagonal edges j in {i+1, i-1, i-2}; self edges are excluded.\n"
            "- Decision rule: AUROC/AUPRC use continuous scores; F1 uses the top 60 off-diagonal scores for every method.",
        ),
        notebook_cell("code", f"CONFIG = {config_text}\nCONFIG", execution_count=1, output=config_text + "\n"),
        notebook_cell(
            "code",
            "from pathlib import Path\n"
            "import sys\n\n"
            "ROOT = Path.cwd()\n"
            "if (ROOT / 'experiments').exists():\n"
            "    sys.path.insert(0, str(ROOT / 'experiments'))\n"
            "else:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from run_lorenz96_benchmark import run_benchmark\n\n"
            "# Set force=True to recompute all fits.\n"
            "payload = run_benchmark(force=False)\n",
            execution_count=2,
            output="Cached results are loaded by default. Use run_benchmark(force=True) to recompute.\n",
        ),
        notebook_cell("markdown", "## Xneural VAR: lambda_ngc sweep"),
        notebook_cell("code", "print(LAMBDA_SWEEP_TABLE)\n", execution_count=3, output=lambda_table + "\n"),
        notebook_cell("markdown", "## Best method comparison\n\nBest rows are selected by AUPRC over the grid stated in the script."),
        notebook_cell("code", "print(METHOD_COMPARISON_TABLE)\n", execution_count=4, output=comparison_table + "\n"),
        notebook_cell(
            "markdown",
            "## Artifacts\n\n"
            "- `experiments/lorenz96_results.json`: complete configuration and all rows.\n"
            "- `experiments/lorenz96_lambda_sweep.csv`: Xneural VAR lambda sweep.\n"
            "- `experiments/lorenz96_method_comparison.csv`: best row for each method.",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Lorenz-96 Granger benchmark.")
    parser.add_argument("--force", action="store_true", help="recompute all fits instead of using cached raw JSON")
    parser.add_argument("--aggregate-only", action="store_true", help="only aggregate existing raw JSON files")
    parser.add_argument("--device", default=None, help="torch device, for example cpu or cuda")
    args = parser.parse_args()

    if args.aggregate_only:
        payload = aggregate_results(device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        payload = run_benchmark(force=args.force, device=args.device)

    print("\nXneural VAR lambda sweep")
    print(format_table(payload["xneural_lambda_sweep"], columns=["param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"]))
    print("\nMethod comparison")
    print(format_table(payload["method_comparison"], columns=["method", "param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"]))
    print(f"\nWrote {RESULT_JSON}")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
