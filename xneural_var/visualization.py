from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Literal

import numpy as np

GateSummaryName = Literal["norm", "max", "mean"]
BoxplotValueName = Literal["signed", "absolute"]


_PAPER_RC = {
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "font.size": 10,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xneural_var visualization functions require matplotlib. "
            "Install it with `pip install matplotlib` or `pip install -e .[viz]`."
        ) from exc
    return plt, TwoSlopeNorm


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_model(result_or_model: Any) -> Any:
    return getattr(result_or_model, "model", result_or_model)


def _get_gate(result_or_model: Any) -> np.ndarray:
    model = _get_model(result_or_model)
    gate = getattr(model, "causal_gate", None)
    if gate is None:
        raise ValueError("plot_causal_gate_by_lag requires an XNeural VAR model with causal_gate.")
    gate_np = _to_numpy(gate)
    if gate_np.ndim != 3:
        raise ValueError("causal_gate must have shape [lag, target, source].")
    return gate_np


def _get_coeffs(result: Any) -> np.ndarray:
    coeffs = getattr(result, "coeffs", None)
    if coeffs is None:
        raise ValueError("plot_edge_lag_boxplots requires result.coeffs from fit_gvar_ngc.")
    coeffs_np = _to_numpy(coeffs)
    if coeffs_np.ndim != 4:
        raise ValueError("result.coeffs must have shape [sample, lag, target, source].")
    return coeffs_np


def _resolve_variable_names(n_vars: int, variable_names: list[str] | tuple[str, ...] | None) -> list[str]:
    if variable_names is None:
        return [f"x{idx}" for idx in range(n_vars)]
    if len(variable_names) != n_vars:
        raise ValueError("variable_names length must match the number of variables.")
    return list(variable_names)


def _lag_titles(order: int) -> list[str]:
    return [f"lag {order - lag_idx}" for lag_idx in range(order)]


def _summary_gate(gate: np.ndarray, summary: GateSummaryName) -> np.ndarray:
    if summary == "norm":
        return np.linalg.norm(gate, axis=0)
    if summary == "max":
        return np.max(np.abs(gate), axis=0)
    if summary == "mean":
        return np.mean(np.abs(gate), axis=0)
    raise ValueError(f"unsupported gate summary: {summary}")


def _heatmap_limits(mats: list[np.ndarray], signed: bool, percentile: float) -> tuple[float, float]:
    values = np.concatenate([np.ravel(mat[np.isfinite(mat)]) for mat in mats])
    if values.size == 0:
        return -1.0, 1.0
    percentile = float(np.clip(percentile, 0.0, 100.0))
    if signed:
        vmax = float(np.percentile(np.abs(values), percentile))
        vmax = max(vmax, np.finfo(float).eps)
        return -vmax, vmax
    vmax = float(np.percentile(values, percentile))
    vmax = max(vmax, np.finfo(float).eps)
    return 0.0, vmax


def _format_ticks(ax: Any, names: list[str]) -> None:
    positions = np.arange(len(names))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(names)
    ax.set_xlabel("source")
    ax.set_ylabel("target")


def _annotate_heatmap(ax: Any, mat: np.ndarray, fmt: str) -> None:
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            ax.text(
                col,
                row,
                format(float(mat[row, col]), fmt),
                ha="center",
                va="center",
                fontsize=7,
                color="#111111",
            )


def _finish_figure(fig: Any, save_path: str | Path | None, dpi: int, show: bool) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)
    if show:
        fig.canvas.draw_idle()


def plot_causal_gate_by_lag(
    result_or_model: Any,
    *,
    variable_names: list[str] | tuple[str, ...] | None = None,
    absolute: bool = True,
    include_summary: bool = True,
    summary: GateSummaryName = "norm",
    percentile: float = 99.0,
    cmap: str | None = None,
    annotate: bool | None = None,
    annotation_format: str = ".2g",
    title: str = "Lag-wise causal gate",
    figsize: tuple[float, float] | None = None,
    ncols: int | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot one causal-gate heatmap per lag for XNeural VAR.

    Parameters
    ----------
    result_or_model:
        A ``FitResult`` returned by ``fit_gvar_ngc`` or a ``GVARWithNGCGates``
        instance with ``use_causal_gate=True``.
    absolute:
        If true, plot ``abs(causal_gate)``. If false, plot signed gate values
        with a diverging colormap.
    include_summary:
        If true, append a final panel summarizing gate strength over lags.
    summary:
        Summary used for the final panel: ``"norm"``, ``"max"``, or ``"mean"``.
    """
    plt, TwoSlopeNorm = _require_matplotlib()
    gate = _get_gate(result_or_model)
    order, n_targets, n_sources = gate.shape
    if n_targets != n_sources:
        raise ValueError("causal_gate must be square in target/source dimensions.")

    names = _resolve_variable_names(n_targets, variable_names)
    signed = not absolute
    gate_to_plot = np.abs(gate) if absolute else gate.copy()

    mats = [gate_to_plot[lag_idx] for lag_idx in range(order)]
    titles = _lag_titles(order)
    if include_summary:
        summary_mat = _summary_gate(gate, summary)
        mats.append(summary_mat)
        titles.append(f"{summary} summary")

    vmin, vmax = _heatmap_limits(mats, signed=signed, percentile=percentile)
    cmap = cmap or ("coolwarm" if signed else "viridis")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if signed else None

    n_panels = len(mats)
    if ncols is None:
        ncols = min(4, n_panels)
    nrows = ceil(n_panels / ncols)
    if figsize is None:
        figsize = (3.2 * ncols + 0.8, 3.1 * nrows + 0.8)
    if annotate is None:
        annotate = n_targets <= 6

    with plt.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.995)
        image = None

        for idx, (mat, panel_title) in enumerate(zip(mats, titles)):
            ax = axes.flat[idx]
            image = ax.imshow(
                mat,
                cmap=cmap,
                vmin=None if signed else vmin,
                vmax=None if signed else vmax,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(panel_title, fontsize=10, fontweight="semibold")
            _format_ticks(ax, names)
            ax.set_xticks(np.arange(-0.5, n_sources, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_targets, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            if annotate:
                _annotate_heatmap(ax, mat, annotation_format)

        for idx in range(n_panels, nrows * ncols):
            axes.flat[idx].axis("off")

        if image is not None:
            cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
            cbar.set_label("|gate|" if absolute else "gate", rotation=270, labelpad=14)

        fig.tight_layout(rect=(0, 0, 0.97, 0.96))
        _finish_figure(fig, save_path, dpi=dpi, show=show)
        return fig, axes


def _strength_from_result_or_coeffs(result: Any, coeffs: np.ndarray) -> np.ndarray:
    strength = getattr(result, "causal_strength", None)
    if strength is not None:
        strength_np = _to_numpy(strength)
        if strength_np.shape == coeffs.shape[2:]:
            return strength_np
    return np.quantile(np.abs(coeffs), 0.95, axis=(0, 1))


def _select_edges(
    result: Any,
    coeffs: np.ndarray,
    edges: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
    top_n: int,
    exclude_self: bool,
) -> list[tuple[int, int]]:
    n_targets, n_sources = coeffs.shape[2:]
    if edges is not None:
        selected = [(int(target), int(source)) for target, source in edges]
        for target, source in selected:
            if not (0 <= target < n_targets and 0 <= source < n_sources):
                raise ValueError("edge indices must be valid (target, source) pairs.")
        return selected

    strength = _strength_from_result_or_coeffs(result, coeffs).copy()
    if exclude_self:
        np.fill_diagonal(strength, -np.inf)
    flat_order = np.argsort(strength.ravel())[::-1]
    selected = []
    for flat_idx in flat_order:
        if len(selected) >= top_n:
            break
        target, source = np.unravel_index(flat_idx, strength.shape)
        if not np.isfinite(strength[target, source]):
            continue
        selected.append((int(target), int(source)))
    return selected


def _boxplot_ylim(values: list[np.ndarray], signed: bool, percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.ravel(value[np.isfinite(value)]) for value in values if value.size])
    if finite.size == 0:
        return -1.0, 1.0
    percentile = float(np.clip(percentile, 0.0, 100.0))
    if signed:
        bound = float(np.percentile(np.abs(finite), percentile))
        bound = max(bound, np.finfo(float).eps)
        return -1.08 * bound, 1.08 * bound
    upper = float(np.percentile(finite, percentile))
    upper = max(upper, np.finfo(float).eps)
    return -0.02 * upper, 1.08 * upper


def _edge_label(target: int, source: int, names: list[str]) -> str:
    return f"{names[source]} -> {names[target]}"


def plot_edge_lag_boxplots(
    result: Any,
    *,
    edges: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    top_n: int = 6,
    exclude_self: bool = True,
    value: BoxplotValueName = "signed",
    variable_names: list[str] | tuple[str, ...] | None = None,
    percentile_ylim: float = 99.0,
    title: str = "Effective coefficient distributions by lag",
    figsize: tuple[float, float] | None = None,
    ncols: int | None = None,
    box_color: str = "#4C78A8",
    median_color: str = "#C43C39",
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot per-edge boxplots with lag on the x-axis and coefficient on the y-axis.

    ``edges`` are specified as ``(target, source)`` pairs, matching the tensor
    convention ``coeffs[:, lag, target, source]``. If ``edges`` is omitted, the
    strongest edges are selected from ``result.causal_strength``.
    """
    _get_gate(result)
    plt, _ = _require_matplotlib()
    coeffs = _get_coeffs(result)
    n_samples, order, n_targets, n_sources = coeffs.shape
    if n_targets != n_sources:
        raise ValueError("coeffs must be square in target/source dimensions.")
    if n_samples == 0:
        raise ValueError("result.coeffs must contain at least one sample.")
    if value not in ("signed", "absolute"):
        raise ValueError("value must be 'signed' or 'absolute'.")

    names = _resolve_variable_names(n_targets, variable_names)
    selected_edges = _select_edges(result, coeffs, edges, top_n=top_n, exclude_self=exclude_self)
    if not selected_edges:
        raise ValueError("no edges selected for plotting.")

    signed = value == "signed"
    edge_values = []
    for target, source in selected_edges:
        lag_values = [coeffs[:, lag_idx, target, source] for lag_idx in range(order)]
        if not signed:
            lag_values = [np.abs(values) for values in lag_values]
        edge_values.extend(lag_values)
    ymin, ymax = _boxplot_ylim(edge_values, signed=signed, percentile=percentile_ylim)

    n_panels = len(selected_edges)
    if ncols is None:
        ncols = min(3, n_panels)
    nrows = ceil(n_panels / ncols)
    if figsize is None:
        figsize = (3.5 * ncols + 0.6, 2.8 * nrows + 0.9)

    xlabels = [str(order - lag_idx) for lag_idx in range(order)]
    ylabel = "effective coefficient" if signed else "|effective coefficient|"

    with plt.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=True)
        fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.995)

        for idx, (target, source) in enumerate(selected_edges):
            ax = axes.flat[idx]
            values_by_lag = [coeffs[:, lag_idx, target, source] for lag_idx in range(order)]
            if not signed:
                values_by_lag = [np.abs(values) for values in values_by_lag]

            box = ax.boxplot(
                values_by_lag,
                patch_artist=True,
                widths=0.58,
                showmeans=True,
                showfliers=False,
                medianprops={"color": median_color, "linewidth": 1.5},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "#222222",
                    "markersize": 3.5,
                },
                whiskerprops={"color": "#333333", "linewidth": 0.9},
                capprops={"color": "#333333", "linewidth": 0.9},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(box_color)
                patch.set_alpha(0.78)
                patch.set_edgecolor("#222222")
                patch.set_linewidth(0.9)

            if signed:
                ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)
            ax.set_ylim(ymin, ymax)
            ax.set_title(_edge_label(target, source, names), fontsize=10, fontweight="semibold")
            ax.set_xticks(np.arange(1, order + 1))
            ax.set_xticklabels(xlabels)
            ax.set_xlabel("lag")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(n_panels, nrows * ncols):
            axes.flat[idx].axis("off")

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _finish_figure(fig, save_path, dpi=dpi, show=show)
        return fig, axes
