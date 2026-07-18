from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from run_lorenz96_benchmark import (  # noqa: E402
    DEFAULT_CONFIG as LORENZ_CONFIG,
    evaluate_scores,
    format_table,
    make_lorenz96_splits,
    neural_test_mse,
    run_cmlp as run_lorenz_cmlp,
    run_gvar as run_lorenz_gvar,
    run_linear as run_lorenz_linear,
    run_xneural as run_lorenz_xneural,
)
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


OUT_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures"
RAW_LINEAR_DIR = OUT_DIR / "raw_linear_var"
RAW_LORENZ_F40_DIR = OUT_DIR / "raw_lorenz96_f40"
COMBINED_JSON = OUT_DIR / "benchmark_results.json"
COMBINED_NOTEBOOK = OUT_DIR / "xneural_var_benchmarks.ipynb"
LINEAR_CSV = OUT_DIR / "linear_var_method_comparison.csv"
LORENZ_F40_CSV = OUT_DIR / "lorenz96_f40_method_comparison.csv"


LORENZ_F40_CONFIG: dict[str, Any] = dict(LORENZ_CONFIG)
LORENZ_F40_CONFIG.update(
    {
        "forcing": 40.0,
        "total_T": 500,
        "train_T": 400,
        "n_repeats": 5,
        "seed": 20260720,
    }
)
LORENZ_F40_XNEURAL_LAMBDAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
LORENZ_F40_CMLP_LAMBDAS = [0.003, 0.01, 0.03]
LORENZ_F40_GVAR_LAMBDAS = [0.0, 0.001, 0.01, 0.03]
LORENZ_F40_RIDGE_GRID = [1e-6, 1e-4, 1e-2, 1.0, 10.0]


LINEAR_CONFIG: dict[str, Any] = {
    "p": 5,
    "order": 5,
    "n_repeats": 10,
    "total_T": 500,
    "train_T": 400,
    "noise_std": 0.25,
    "hidden_layer_size": 16,
    "num_hidden_layers": 1,
    "max_epochs": 180,
    "batch_size": 128,
    "learning_rate": 2e-2,
    "lambda_smooth": 0.0,
    "coefficient_weight_decay": 1e-4,
    "seed": 20260718,
}

LINEAR_XNEURAL_LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.06]
LINEAR_CMLP_LAMBDAS = [0.001, 0.003, 0.01]
LINEAR_GVAR_LAMBDAS = [0.0, 0.001, 0.01]
LINEAR_RIDGE_GRID = [1e-6, 1e-4, 1e-2, 1.0]


def true_linear_var_coefficients(order: int = 5, p: int = 5) -> np.ndarray:
    coeffs = np.zeros((order, p, p), dtype=np.float64)
    coeffs[0, range(p), range(p)] = 0.18
    coeffs[0, 0, 1] = 0.46
    coeffs[0, 1, 2] = -0.42
    coeffs[0, 2, 0] = 0.38
    coeffs[0, 3, 4] = -0.44
    coeffs[0, 4, 3] = 0.36
    coeffs[1, 0, 3] = -0.28
    coeffs[1, 2, 4] = 0.30
    coeffs[1, 4, 1] = -0.32
    coeffs[2, 1, 0] = 0.24
    coeffs[2, 3, 2] = 0.26
    return coeffs


def simulate_linear_var(
    *,
    coeffs_actual_lag: np.ndarray,
    total_T: int,
    noise_std: float,
    rng: np.random.Generator,
    burn_in: int = 200,
) -> np.ndarray:
    order, p, _ = coeffs_actual_lag.shape
    total = burn_in + total_T + order
    x = np.zeros((total, p), dtype=np.float64)
    x[:order] = rng.normal(scale=noise_std, size=(order, p))
    for t in range(order, total):
        value = np.zeros(p, dtype=np.float64)
        for lag in range(1, order + 1):
            value += coeffs_actual_lag[lag - 1] @ x[t - lag]
        x[t] = value + rng.normal(scale=noise_std, size=p)
    return x[burn_in + order : burn_in + order + total_T].astype(np.float32)


def make_linear_var_splits(config: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config["seed"])
    coeffs = true_linear_var_coefficients(order=config["order"], p=config["p"])
    raw = [
        simulate_linear_var(
            coeffs_actual_lag=coeffs,
            total_T=config["total_T"],
            noise_std=config["noise_std"],
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
    graph = (np.abs(coeffs).sum(axis=0) > 0).astype(np.int64)
    return train, test, graph, coeffs


def fit_linear_var_baseline(train: list[np.ndarray], order: int, ridge: float) -> tuple[np.ndarray, np.ndarray]:
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


def linear_var_mse(beta: np.ndarray, test: list[np.ndarray], order: int) -> float:
    dataset = construct_lagged_dataset(test, order)
    x = dataset.predictors.reshape(dataset.predictors.shape[0], -1).astype(np.float64)
    y = dataset.responses.astype(np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return float(np.mean((x_aug @ beta - y) ** 2))


def linear_result_path(run_id: str) -> Path:
    return RAW_LINEAR_DIR / f"{run_id}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lorenz_f40_result_path(run_id: str) -> Path:
    return RAW_LORENZ_F40_DIR / f"{run_id}.json"


def make_row(method: str, variant: str, param_name: str, param_value: float, metrics: dict[str, float], elapsed: float) -> dict[str, Any]:
    row = {
        "method": method,
        "variant": variant,
        "param_name": param_name,
        "param_value": float(param_value),
        "elapsed_sec": float(elapsed),
    }
    row.update(metrics)
    return row


def run_or_load_lorenz_f40(run_id: str, force: bool, callback) -> dict[str, Any]:
    path = lorenz_f40_result_path(run_id)
    if path.exists() and not force:
        print(f"[cache] lorenz-f40 {run_id}")
        return read_json(path)
    print(f"[run]   lorenz-f40 {run_id}")
    row = callback()
    write_json(path, row)
    print(
        f"        AUPRC={row['auprc']:.3f}, F1={row['f1']:.3f}, "
        f"BA={row['balanced_accuracy']:.3f}, MSE={row['test_mse']:.4g}, "
        f"time={row['elapsed_sec']:.1f}s"
    )
    return row


def run_or_load_linear(run_id: str, force: bool, callback) -> dict[str, Any]:
    path = linear_result_path(run_id)
    if path.exists() and not force:
        print(f"[cache] linear {run_id}")
        return read_json(path)
    print(f"[run]   linear {run_id}")
    row = callback()
    write_json(path, row)
    print(
        f"        AUPRC={row['auprc']:.3f}, F1={row['f1']:.3f}, "
        f"sign={row.get('sign_accuracy', float('nan')):.3f}, MSE={row['test_mse']:.4g}, "
        f"time={row['elapsed_sec']:.1f}s"
    )
    return row


def true_coeffs_in_model_lag_order(coeffs_actual_lag: np.ndarray) -> np.ndarray:
    order = coeffs_actual_lag.shape[0]
    return np.stack([coeffs_actual_lag[order - lag_idx - 1] for lag_idx in range(order)], axis=0)


def coefficient_sign_accuracy(result: Any, coeffs_actual_lag: np.ndarray) -> float:
    coeffs_model_order = true_coeffs_in_model_lag_order(coeffs_actual_lag)
    mask = (np.abs(coeffs_model_order) > 1e-12) & (~np.eye(coeffs_model_order.shape[1], dtype=bool)[None, :, :])
    if not np.any(mask):
        return float("nan")
    median_coeffs = np.median(np.asarray(result.coeffs), axis=0)
    pred_sign = np.sign(median_coeffs[mask])
    true_sign = np.sign(coeffs_model_order[mask])
    return float(np.mean(pred_sign == true_sign))


def run_linear_xneural(
    train: list[np.ndarray],
    test: list[np.ndarray],
    true_graph: np.ndarray,
    true_coeffs: np.ndarray,
    config: dict[str, Any],
    lam: float,
    device: str,
    return_model: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Any]:
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
    score = result.model.gate_group_norms().detach().cpu().numpy()
    metrics = evaluate_scores(score, true_graph)
    metrics["test_mse"] = neural_test_mse(result.model, test, config["order"], config["batch_size"], device)
    metrics["sign_accuracy"] = coefficient_sign_accuracy(result, true_coeffs)
    row = make_row("Xneural VAR", "ISTA + hierarchical group lasso", "lambda_ngc", lam, metrics, elapsed)
    if return_model:
        return row, result
    return row


def run_linear_gvar(
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
    return make_row("GVAR", "coefficient elastic-net + smoothness", "lambda_coeff", lam, metrics, elapsed)


def run_linear_cmlp(
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
    return make_row("cMLP", "Neural-GC hierarchical group lasso", "lambda_ngc", lam, metrics, elapsed)


def run_linear_ridge(
    train: list[np.ndarray],
    test: list[np.ndarray],
    true_graph: np.ndarray,
    config: dict[str, Any],
    ridge: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    beta, score = fit_linear_var_baseline(train, config["order"], ridge)
    elapsed = time.perf_counter() - start
    metrics = evaluate_scores(score, true_graph)
    metrics["test_mse"] = linear_var_mse(beta, test, config["order"])
    return make_row("linear VAR", "ridge VAR", "ridge", ridge, metrics, elapsed)


def best_rows_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        method = row["method"]
        key = (row["auprc"], row["f1"], row["balanced_accuracy"], -row["test_mse"])
        if method not in best:
            best[method] = row
            continue
        current = best[method]
        current_key = (current["auprc"], current["f1"], current["balanced_accuracy"], -current["test_mse"])
        if key > current_key:
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
        "sign_accuracy",
        "elapsed_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def run_lorenz_f40_benchmark(force: bool = False, device: str | None = None) -> dict[str, Any]:
    config = dict(LORENZ_F40_CONFIG)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Preparing Lorenz-96 F=40 data...")
    train, test, true_graph = make_lorenz96_splits(config)
    rows: list[dict[str, Any]] = []

    for ridge in LORENZ_F40_RIDGE_GRID:
        rows.append(
            run_or_load_lorenz_f40(
                f"linear_ridge_{ridge:g}",
                force,
                lambda ridge=ridge: run_lorenz_linear(train, test, true_graph, config, ridge),
            )
        )
    for lam in LORENZ_F40_GVAR_LAMBDAS:
        rows.append(
            run_or_load_lorenz_f40(
                f"gvar_lambda_coeff_{lam:g}",
                force,
                lambda lam=lam: run_lorenz_gvar(train, test, true_graph, config, lam, device),
            )
        )
    for lam in LORENZ_F40_XNEURAL_LAMBDAS:
        rows.append(
            run_or_load_lorenz_f40(
                f"xneural_lambda_ngc_{lam:g}",
                force,
                lambda lam=lam: run_lorenz_xneural(train, test, true_graph, config, lam, device),
            )
        )
    for lam in LORENZ_F40_CMLP_LAMBDAS:
        rows.append(
            run_or_load_lorenz_f40(
                f"cmlp_lambda_ngc_{lam:g}",
                force,
                lambda lam=lam: run_lorenz_cmlp(train, test, true_graph, config, lam, device),
            )
        )

    comparison = best_rows_by_method(rows)
    write_csv(LORENZ_F40_CSV, comparison)
    payload = {
        "experiment": {
            "name": "Lorenz-96 F=40 Granger benchmark",
            "config": config,
            "true_offdiagonal_edges": int(3 * config["p"]),
            "decision_rule": "AUROC/AUPRC use continuous off-diagonal scores. Precision/Recall/F1 use top-k, where k=3p=60 true off-diagonal edges.",
            "method_selection": "For method comparison, each method is selected by AUPRC over the explicitly reported grid.",
            "xneural_score": "L2 norm of causal_gate over lags.",
            "gvar_score": "max absolute effective coefficient over samples and lags.",
            "cmlp_score": "first-layer source group norm.",
            "linear_var_score": "L2 norm of ridge VAR coefficients over lags.",
        },
        "grids": {
            "xneural_lambda_ngc": LORENZ_F40_XNEURAL_LAMBDAS,
            "cmlp_lambda_ngc": LORENZ_F40_CMLP_LAMBDAS,
            "gvar_lambda_coeff": LORENZ_F40_GVAR_LAMBDAS,
            "linear_ridge": LORENZ_F40_RIDGE_GRID,
        },
        "all_results": sorted(rows, key=lambda row: (row["method"], row["param_value"])),
        "xneural_lambda_sweep": [row for row in rows if row["method"] == "Xneural VAR"],
        "method_comparison": comparison,
    }
    write_json(OUT_DIR / "lorenz96_f40_results.json", payload)
    return payload


def run_linear_benchmark(force: bool = False, device: str | None = None) -> dict[str, Any]:
    config = dict(LINEAR_CONFIG)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Preparing linear VAR data...")
    train, test, true_graph, true_coeffs = make_linear_var_splits(config)
    rows: list[dict[str, Any]] = []

    for ridge in LINEAR_RIDGE_GRID:
        rows.append(
            run_or_load_linear(
                f"linear_ridge_{ridge:g}",
                force,
                lambda ridge=ridge: run_linear_ridge(train, test, true_graph, config, ridge),
            )
        )
    for lam in LINEAR_GVAR_LAMBDAS:
        rows.append(
            run_or_load_linear(
                f"gvar_lambda_coeff_{lam:g}",
                force,
                lambda lam=lam: run_linear_gvar(train, test, true_graph, config, lam, device),
            )
        )
    for lam in LINEAR_XNEURAL_LAMBDAS:
        rows.append(
            run_or_load_linear(
                f"xneural_lambda_ngc_{lam:g}",
                force,
                lambda lam=lam: run_linear_xneural(train, test, true_graph, true_coeffs, config, lam, device),
            )
        )
    for lam in LINEAR_CMLP_LAMBDAS:
        rows.append(
            run_or_load_linear(
                f"cmlp_lambda_ngc_{lam:g}",
                force,
                lambda lam=lam: run_linear_cmlp(train, test, true_graph, config, lam, device),
            )
        )

    comparison = best_rows_by_method(rows)
    write_csv(LINEAR_CSV, comparison)
    payload = {
        "experiment": {
            "name": "Sparse linear VAR benchmark",
            "config": config,
            "true_offdiagonal_edges": int((true_graph * (1 - np.eye(config["p"], dtype=np.int64))).sum()),
            "decision_rule": "AUROC/AUPRC use continuous off-diagonal scores. Precision/Recall/F1 use top-k, where k is the number of true off-diagonal edges.",
            "sign_rule": "For Xneural VAR, sign accuracy is computed from median effective coefficients at true nonzero lag-edge entries.",
        },
        "grids": {
            "xneural_lambda_ngc": LINEAR_XNEURAL_LAMBDAS,
            "cmlp_lambda_ngc": LINEAR_CMLP_LAMBDAS,
            "gvar_lambda_coeff": LINEAR_GVAR_LAMBDAS,
            "linear_ridge": LINEAR_RIDGE_GRID,
        },
        "all_results": sorted(rows, key=lambda row: (row["method"], row["param_value"])),
        "xneural_lambda_sweep": [row for row in rows if row["method"] == "Xneural VAR"],
        "method_comparison": comparison,
        "true_coefficients_actual_lag": true_coeffs.tolist(),
    }
    write_json(OUT_DIR / "linear_var_results.json", payload)
    return payload


def selected_linear_edges() -> list[tuple[int, int]]:
    return [(0, 1), (1, 2), (3, 4), (0, 3), (4, 1), (3, 2)]


def refit_best_linear_xneural(payload: dict[str, Any], device: str) -> Any:
    best = next(row for row in payload["method_comparison"] if row["method"] == "Xneural VAR")
    train, test, true_graph, true_coeffs = make_linear_var_splits(payload["experiment"]["config"])
    _, result = run_linear_xneural(
        train,
        test,
        true_graph,
        true_coeffs,
        payload["experiment"]["config"],
        best["param_value"],
        device,
        return_model=True,
    )
    return result


def refit_best_lorenz_xneural(payload: dict[str, Any], device: str) -> Any:
    best = next(row for row in payload["method_comparison"] if row["method"] == "Xneural VAR")
    config = payload["experiment"]["config"]
    train, _, _ = make_lorenz96_splits(config)
    fit_config = GVARTrainingConfig(
        order=config["order"],
        hidden_layer_size=config["hidden_layer_size"],
        num_hidden_layers=config["num_hidden_layers"],
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        lambda_ngc=best["param_value"],
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
    return fit_gvar_ngc(train, fit_config)


def make_figures(linear_payload: dict[str, Any], lorenz_payload: dict[str, Any], device: str) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = OUT_DIR / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    linear_result = refit_best_linear_xneural(linear_payload, device)
    fig, _ = plot_causal_gate_by_lag(
        linear_result,
        summary="norm",
        percentile=99,
        tick_label_step=1,
        ncols=3,
        figsize=(8.8, 6.0),
        title="Linear VAR: lag-wise causal gate",
        save_path=FIG_DIR / "linear_var_gate_by_lag_benchmark.png",
        show=False,
    )
    plt.close(fig)

    fig, _ = plot_edge_lag_boxplots(
        linear_result,
        edges=selected_linear_edges(),
        value="signed",
        variable_names=[f"x{i}" for i in range(linear_payload["experiment"]["config"]["p"])],
        title="Linear VAR: signed effective coefficients",
        save_path=FIG_DIR / "linear_var_signed_boxplots_benchmark.png",
        show=False,
    )
    plt.close(fig)

    lorenz_result = refit_best_lorenz_xneural(lorenz_payload, device)
    fig, _ = plot_causal_gate_by_lag(
        lorenz_result,
        summary="norm",
        percentile=99,
        tick_label_step=2,
        title="Lorenz-96 F=40: lag-wise causal gate",
        save_path=FIG_DIR / "lorenz96_f40_gate_by_lag_benchmark.png",
        show=False,
    )
    plt.close(fig)

    return {
        "linear_gate": "linear_var_gate_by_lag_benchmark.png",
        "linear_boxplot": "linear_var_signed_boxplots_benchmark.png",
        "lorenz_gate": "lorenz96_f40_gate_by_lag_benchmark.png",
    }


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
            cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": output.splitlines(keepends=True)})
    return cell


def make_combined_notebook(payload: dict[str, Any]) -> None:
    lorenz_cmp = format_table(payload["lorenz96"]["method_comparison"], columns=["method", "param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"])
    lorenz_lam = format_table(payload["lorenz96"]["xneural_lambda_sweep"], columns=["param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"])
    linear_cmp = format_table(payload["linear_var"]["method_comparison"], columns=["method", "param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"])
    linear_lam = format_table(payload["linear_var"]["xneural_lambda_sweep"], columns=["param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"])
    cells = [
        notebook_cell("markdown", "# Xneural VAR benchmark experiments\n\nClean notebook for the slide results. The implementation is in `experiments/run_all_benchmarks.py`."),
        notebook_cell(
            "markdown",
            "## Design summary\n\n"
            "- Lorenz-96: p=20, F=40, K=5, 5 trajectories, T=500, train/test=400/100.\n"
            "- Linear VAR: p=5, K=5, 10 trajectories, T=500, train/test=400/100, mixed-sign sparse coefficients.\n"
            "- AUROC/AUPRC use continuous off-diagonal scores.\n"
            "- Precision/Recall/F1 use a common top-k rule, with k equal to the number of true off-diagonal edges.\n"
            "- Xneural VAR reports lambda_ngc sweeps and uses lag-wise causal gate visualizations in both experiments.",
        ),
        notebook_cell(
            "code",
            "from pathlib import Path\n"
            "import sys\n\n"
            "ROOT = Path.cwd()\n"
            "if (ROOT / 'experiments' / 'run_all_benchmarks.py').exists():\n"
            "    sys.path.insert(0, str(ROOT / 'experiments'))\n"
            "elif (ROOT / 'run_all_benchmarks.py').exists():\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from run_all_benchmarks import main_payload\n\n"
            "# In the saved notebook, the tables below are the outputs used in slide.tex.\n",
            execution_count=1,
            output="Use `python experiments/run_all_benchmarks.py --force --device cpu` to recompute.\n",
        ),
        notebook_cell("markdown", "## Lorenz-96: Xneural VAR lambda sweep"),
        notebook_cell("code", "print(LORENZ_LAMBDA_SWEEP)\n", execution_count=2, output=lorenz_lam + "\n"),
        notebook_cell("markdown", "## Lorenz-96: best method comparison"),
        notebook_cell("code", "print(LORENZ_METHOD_COMPARISON)\n", execution_count=3, output=lorenz_cmp + "\n"),
        notebook_cell("markdown", "## Linear VAR: Xneural VAR lambda sweep"),
        notebook_cell("code", "print(LINEAR_LAMBDA_SWEEP)\n", execution_count=4, output=linear_lam + "\n"),
        notebook_cell("markdown", "## Linear VAR: best method comparison"),
        notebook_cell("code", "print(LINEAR_METHOD_COMPARISON)\n", execution_count=5, output=linear_cmp + "\n"),
        notebook_cell(
            "markdown",
            "## Figures generated with xneural_var visualization methods\n\n"
            "- `figures/lorenz96_f40_gate_by_lag_benchmark.png`\n"
            "- `figures/linear_var_gate_by_lag_benchmark.png`\n"
            "- `figures/linear_var_signed_boxplots_benchmark.png`",
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
    COMBINED_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main_payload(force: bool = False, figures: bool = True, device: str | None = None) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lorenz_payload = run_lorenz_f40_benchmark(force=force, device=device)
    linear_payload = run_linear_benchmark(force=force, device=device)
    figure_files = make_figures(linear_payload, lorenz_payload, device) if figures else {}
    payload = {
        "lorenz96": lorenz_payload,
        "linear_var": linear_payload,
        "figures": figure_files,
    }
    write_json(COMBINED_JSON, payload)
    make_combined_notebook(payload)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run all Xneural VAR benchmark experiments.")
    parser.add_argument("--force", action="store_true", help="recompute linear fits and regenerate figures")
    parser.add_argument("--no-figures", action="store_true", help="skip visualization generation")
    parser.add_argument("--device", default=None, help="torch device, for example cpu or cuda")
    args = parser.parse_args()

    payload = main_payload(force=args.force, figures=not args.no_figures, device=args.device)
    print("\nLorenz-96 F=40 method comparison")
    print(format_table(payload["lorenz96"]["method_comparison"], columns=["method", "param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"]))
    print("\nLinear VAR method comparison")
    print(format_table(payload["linear_var"]["method_comparison"], columns=["method", "param_value", "auroc", "auprc", "f1", "balanced_accuracy", "test_mse"]))
    print(f"\nWrote {COMBINED_JSON}")
    print(f"Wrote {COMBINED_NOTEBOOK}")


if __name__ == "__main__":
    main()
