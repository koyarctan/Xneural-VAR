from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

RegularizerName = Literal["none", "sparse_group_lasso", "group_lasso", "hierarchical_group_lasso"]
ReductionName = Literal["sum", "mean"]


def _group_norms_by_pair(tensor: torch.Tensor, lag_dim: int) -> torch.Tensor:
    dims = tuple(dim for dim in range(tensor.ndim) if dim not in (lag_dim, tensor.ndim - 2, tensor.ndim - 1))
    if dims:
        tensor = torch.linalg.vector_norm(tensor, ord=2, dim=dims)
        if lag_dim > 0:
            lag_dim = 0
    return torch.linalg.vector_norm(tensor, ord=2, dim=lag_dim)


def _reduce(values: torch.Tensor, reduction: ReductionName) -> torch.Tensor:
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def group_lasso_penalty(
    tensor: torch.Tensor,
    *,
    lag_dim: int = 0,
    reduction: ReductionName = "sum",
) -> torch.Tensor:
    """Group lasso over lag blocks for every target-source pair.

    Supports gate tensors ``[lag, target, source]`` and coefficient tensors
    ``[batch, lag, target, source]``.
    """
    norms = _group_norms_by_pair(tensor, lag_dim=lag_dim)
    return _reduce(norms, reduction)


def sparse_group_lasso_penalty(
    tensor: torch.Tensor,
    *,
    lag_dim: int = 0,
    reduction: ReductionName = "sum",
    l1_weight: float = 1.0,
    group_weight: float = 1.0,
) -> torch.Tensor:
    """Sparse group lasso over lag blocks for every target-source pair.

    Neural-GC's cMLP implements ``GSGL`` as lag-level shrinkage followed by
    source-level group shrinkage. For the gate tensor used here, each lag-level
    gate is a scalar, so the lag-level term becomes an L1 penalty while the
    edge-level term is the usual group lasso across lags.
    """
    penalty = tensor.new_zeros(())
    if group_weight:
        penalty = penalty + group_weight * group_lasso_penalty(
            tensor,
            lag_dim=lag_dim,
            reduction=reduction,
        )
    if l1_weight:
        penalty = penalty + l1_weight * _reduce(tensor.abs(), reduction)
    return penalty


def hierarchical_group_lasso_penalty(
    tensor: torch.Tensor,
    *,
    lag_dim: int = 0,
    reduction: ReductionName = "sum",
) -> torch.Tensor:
    """Nested lag-prefix group lasso.

    Lag index 0 is assumed to be the most distant lag, matching the GVAR data
    layout and the Neural-GC cMLP hierarchical penalty convention.
    """
    tensor = tensor.movedim(lag_dim, 0)
    penalties = [
        group_lasso_penalty(tensor[: lag_idx + 1], lag_dim=0, reduction=reduction)
        for lag_idx in range(tensor.shape[0])
    ]
    return torch.stack(penalties).sum()


@torch.no_grad()
def prox_lasso_(param: torch.Tensor, lam: float, step_size: float) -> torch.Tensor:
    """In-place elementwise soft-thresholding."""
    threshold = lam * step_size
    param.copy_(param.sign() * torch.clamp(param.abs() - threshold, min=0.0))
    return param


@torch.no_grad()
def prox_group_lasso_(param: torch.Tensor, lam: float, step_size: float, eps: float = 1e-12) -> torch.Tensor:
    """In-place proximal operator for group lasso on ``[lag, target, source]``."""
    if param.ndim != 3:
        raise ValueError("prox_group_lasso_ expects [lag, target, source]")
    threshold = lam * step_size
    norms = torch.linalg.vector_norm(param, ord=2, dim=0, keepdim=True)
    scale = torch.clamp(1.0 - threshold / torch.clamp(norms, min=eps), min=0.0)
    param.mul_(scale)
    param.masked_fill_(norms <= threshold, 0.0)
    return param


@torch.no_grad()
def prox_sparse_group_lasso_(
    param: torch.Tensor,
    lam: float,
    step_size: float,
    *,
    l1_weight: float = 1.0,
    group_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """In-place proximal operator for sparse group lasso.

    The update mirrors Neural-GC's ``GSGL`` order: first apply within-group
    sparsity, then shrink each target-source lag vector as one group.
    """
    if param.ndim != 3:
        raise ValueError("prox_sparse_group_lasso_ expects [lag, target, source]")
    if l1_weight:
        prox_lasso_(param, lam * l1_weight, step_size)
    if group_weight:
        prox_group_lasso_(param, lam * group_weight, step_size, eps=eps)
    return param


@torch.no_grad()
def prox_hierarchical_group_lasso_(
    param: torch.Tensor,
    lam: float,
    step_size: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """In-place nested proximal update for hierarchical group lasso.

    This follows the iterative nested-group shrinkage used in Neural-GC: each
    prefix ``param[:lag_idx + 1]`` is shrunk as one group for every
    target-source pair.
    """
    if param.ndim != 3:
        raise ValueError("prox_hierarchical_group_lasso_ expects [lag, target, source]")
    threshold = lam * step_size
    for lag_idx in range(param.shape[0]):
        block = param[: lag_idx + 1]
        norms = torch.linalg.vector_norm(block, ord=2, dim=0, keepdim=True)
        scale = torch.clamp(1.0 - threshold / torch.clamp(norms, min=eps), min=0.0)
        block.mul_(scale)
        block.masked_fill_(norms <= threshold, 0.0)
    return param


@dataclass(frozen=True)
class NGCRegularizer:
    name: RegularizerName = "sparse_group_lasso"
    lam: float = 0.0
    reduction: ReductionName = "sum"
    lag_dim: int = 0
    sparse_l1_weight: float = 1.0
    sparse_group_weight: float = 1.0

    def penalty(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.name == "none" or self.lam == 0:
            return tensor.new_zeros(())
        if self.name == "sparse_group_lasso":
            return self.lam * sparse_group_lasso_penalty(
                tensor,
                lag_dim=self.lag_dim,
                reduction=self.reduction,
                l1_weight=self.sparse_l1_weight,
                group_weight=self.sparse_group_weight,
            )
        if self.name == "group_lasso":
            return self.lam * group_lasso_penalty(
                tensor,
                lag_dim=self.lag_dim,
                reduction=self.reduction,
            )
        if self.name == "hierarchical_group_lasso":
            return self.lam * hierarchical_group_lasso_penalty(
                tensor,
                lag_dim=self.lag_dim,
                reduction=self.reduction,
            )
        raise ValueError(f"unsupported regularizer: {self.name}")

    @torch.no_grad()
    def prox_(self, param: torch.Tensor, step_size: float) -> torch.Tensor:
        if self.name == "none" or self.lam == 0:
            return param
        if self.lag_dim != 0:
            raise ValueError("prox_ expects lag_dim=0 and param shape [lag, target, source]")
        if self.name == "sparse_group_lasso":
            return prox_sparse_group_lasso_(
                param,
                self.lam,
                step_size,
                l1_weight=self.sparse_l1_weight,
                group_weight=self.sparse_group_weight,
            )
        if self.name == "group_lasso":
            return prox_group_lasso_(param, self.lam, step_size)
        if self.name == "hierarchical_group_lasso":
            return prox_hierarchical_group_lasso_(param, self.lam, step_size)
        raise ValueError(f"unsupported regularizer: {self.name}")
