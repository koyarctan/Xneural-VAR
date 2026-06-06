from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

RegularizerName = Literal["none", "group_lasso", "hierarchical_group_lasso"]
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
    name: RegularizerName = "group_lasso"
    lam: float = 0.0
    reduction: ReductionName = "sum"
    lag_dim: int = 0

    def penalty(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.name == "none" or self.lam == 0:
            return tensor.new_zeros(())
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
        if self.name == "group_lasso":
            return prox_group_lasso_(param, self.lam, step_size)
        if self.name == "hierarchical_group_lasso":
            return prox_hierarchical_group_lasso_(param, self.lam, step_size)
        raise ValueError(f"unsupported regularizer: {self.name}")
