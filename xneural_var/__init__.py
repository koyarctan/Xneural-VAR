from importlib import import_module

from .data import LaggedDataset, construct_lagged_dataset

__all__ = [
    "CMLP",
    "CMLPFitResult",
    "CMLPTrainingConfig",
    "FitResult",
    "GVARBaselineFitResult",
    "GVARBaselineTrainingConfig",
    "GVARTrainingConfig",
    "GVARWithNGCGates",
    "LaggedDataset",
    "NGCRegularizer",
    "RegularizerName",
    "construct_lagged_dataset",
    "fit_cmlp",
    "fit_gvar",
    "fit_gvar_ngc",
    "group_lasso_penalty",
    "hierarchical_group_lasso_penalty",
    "plot_causal_gate_by_lag",
    "plot_edge_lag_boxplots",
    "prox_lasso_",
    "prox_group_lasso_",
    "prox_hierarchical_group_lasso_",
    "prox_sparse_group_lasso_",
    "sparse_group_lasso_penalty",
]

_LAZY_ATTRS = {
    "CMLP": ("xneural_var.cmlp", "CMLP"),
    "CMLPFitResult": ("xneural_var.cmlp", "CMLPFitResult"),
    "CMLPTrainingConfig": ("xneural_var.cmlp", "CMLPTrainingConfig"),
    "FitResult": ("xneural_var.training", "FitResult"),
    "GVARBaselineFitResult": ("xneural_var.gvar", "GVARBaselineFitResult"),
    "GVARBaselineTrainingConfig": ("xneural_var.gvar", "GVARBaselineTrainingConfig"),
    "GVARTrainingConfig": ("xneural_var.training", "GVARTrainingConfig"),
    "GVARWithNGCGates": ("xneural_var.models", "GVARWithNGCGates"),
    "NGCRegularizer": ("xneural_var.regularizers", "NGCRegularizer"),
    "RegularizerName": ("xneural_var.regularizers", "RegularizerName"),
    "fit_cmlp": ("xneural_var.cmlp", "fit_cmlp"),
    "fit_gvar": ("xneural_var.gvar", "fit_gvar"),
    "fit_gvar_ngc": ("xneural_var.training", "fit_gvar_ngc"),
    "group_lasso_penalty": ("xneural_var.regularizers", "group_lasso_penalty"),
    "hierarchical_group_lasso_penalty": (
        "xneural_var.regularizers",
        "hierarchical_group_lasso_penalty",
    ),
    "plot_causal_gate_by_lag": ("xneural_var.visualization", "plot_causal_gate_by_lag"),
    "plot_edge_lag_boxplots": ("xneural_var.visualization", "plot_edge_lag_boxplots"),
    "prox_lasso_": ("xneural_var.regularizers", "prox_lasso_"),
    "prox_group_lasso_": ("xneural_var.regularizers", "prox_group_lasso_"),
    "prox_hierarchical_group_lasso_": (
        "xneural_var.regularizers",
        "prox_hierarchical_group_lasso_",
    ),
    "prox_sparse_group_lasso_": ("xneural_var.regularizers", "prox_sparse_group_lasso_"),
    "sparse_group_lasso_penalty": ("xneural_var.regularizers", "sparse_group_lasso_penalty"),
}


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
