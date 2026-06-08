from importlib import import_module

from .data import LaggedDataset, construct_lagged_dataset

__all__ = [
    "FitResult",
    "GVARTrainingConfig",
    "GVARWithNGCGates",
    "LaggedDataset",
    "NGCRegularizer",
    "RegularizerName",
    "construct_lagged_dataset",
    "fit_gvar_ngc",
    "group_lasso_penalty",
    "hierarchical_group_lasso_penalty",
    "prox_lasso_",
    "prox_group_lasso_",
    "prox_hierarchical_group_lasso_",
    "prox_sparse_group_lasso_",
    "sparse_group_lasso_penalty",
]

_LAZY_ATTRS = {
    "FitResult": ("xneural_var.training", "FitResult"),
    "GVARTrainingConfig": ("xneural_var.training", "GVARTrainingConfig"),
    "GVARWithNGCGates": ("xneural_var.models", "GVARWithNGCGates"),
    "NGCRegularizer": ("xneural_var.regularizers", "NGCRegularizer"),
    "RegularizerName": ("xneural_var.regularizers", "RegularizerName"),
    "fit_gvar_ngc": ("xneural_var.training", "fit_gvar_ngc"),
    "group_lasso_penalty": ("xneural_var.regularizers", "group_lasso_penalty"),
    "hierarchical_group_lasso_penalty": (
        "xneural_var.regularizers",
        "hierarchical_group_lasso_penalty",
    ),
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
