from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn

from .data import LaggedDataset, construct_lagged_dataset
from .models import AggregationName, GVARWithNGCGates
from .regularizers import NGCRegularizer, RegularizerName

OptimizerName = Literal["adam", "ista"]
SmoothnessMode = Literal["absolute", "relative"]


@dataclass(frozen=True)
class GVARTrainingConfig:
    order: int
    hidden_layer_size: int
    num_hidden_layers: int = 1
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    lambda_ngc: float = 0.0
    lambda_smooth: float = 0.0
    coefficient_weight_decay: float = 0.0
    regularizer: RegularizerName = "group_lasso"
    optimizer: OptimizerName = "ista"
    gate_init: float = 1.0
    seed: int | None = 42
    shuffle: bool = True
    causal_threshold: float = 1e-8
    strength_aggregation: AggregationName = "max"
    smoothness_mode: SmoothnessMode = "absolute"
    smoothness_eps: float = 1e-8
    device: str | torch.device | None = None


@dataclass
class FitResult:
    model: GVARWithNGCGates
    history: dict[str, list[float]] = field(default_factory=dict)
    coeffs: np.ndarray | None = None
    causal_strength: np.ndarray | None = None
    causal_graph: np.ndarray | None = None


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_optimizer(config: GVARTrainingConfig, model: nn.Module) -> torch.optim.Optimizer:
    gate_params = []
    coeff_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "causal_gate":
            gate_params.append(param)
        else:
            coeff_params.append(param)

    param_groups = [
        {"params": coeff_params, "weight_decay": config.coefficient_weight_decay},
        {"params": gate_params, "weight_decay": 0.0},
    ]
    if config.optimizer == "adam":
        return torch.optim.Adam(param_groups, lr=config.learning_rate)
    if config.optimizer == "ista":
        return torch.optim.SGD(param_groups, lr=config.learning_rate)
    raise ValueError(f"unsupported optimizer: {config.optimizer}")


def _iter_batches(n_samples: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        yield indices[start : start + batch_size]


def temporal_smoothness_penalty(
    coeffs: torch.Tensor,
    time_index: torch.Tensor,
    mode: SmoothnessMode = "absolute",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalty for adjacent coefficient changes within the same continuous run."""
    if coeffs.shape[0] < 2:
        return coeffs.new_zeros(())

    order = torch.argsort(time_index)
    sorted_t = time_index[order]
    sorted_coeffs = coeffs[order]
    adjacent = (sorted_t[1:] - sorted_t[:-1]) == 1
    if not torch.any(adjacent):
        return coeffs.new_zeros(())
    diffs = sorted_coeffs[1:][adjacent] - sorted_coeffs[:-1][adjacent]
    if mode == "absolute":
        return torch.mean(diffs.pow(2))
    if mode == "relative":
        denom = sorted_coeffs[:-1][adjacent].pow(2).mean(dim=(1, 2, 3), keepdim=True) + eps
        return torch.mean(diffs.pow(2) / denom)
    raise ValueError(f"unsupported smoothness mode: {mode}")


def _epoch(
    model: GVARWithNGCGates,
    dataset: LaggedDataset,
    config: GVARTrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    ngc: NGCRegularizer,
    criterion: nn.Module,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "ngc": 0.0,
        "smooth": 0.0,
    }
    n_batches = 0

    for batch_idx in _iter_batches(
        dataset.predictors.shape[0],
        config.batch_size,
        shuffle=config.shuffle and train,
        rng=rng,
    ):
        inputs = torch.as_tensor(dataset.predictors[batch_idx], dtype=torch.float32, device=device)
        targets = torch.as_tensor(dataset.responses[batch_idx], dtype=torch.float32, device=device)
        time_index = torch.as_tensor(dataset.time_index[batch_idx], dtype=torch.long, device=device)

        preds, coeffs = model(inputs)
        mse = criterion(preds, targets)
        smooth = config.lambda_smooth * temporal_smoothness_penalty(
            coeffs,
            time_index,
            mode=config.smoothness_mode,
            eps=config.smoothness_eps,
        )

        if model.causal_gate is None:
            raise RuntimeError("NGC regularization requires use_causal_gate=True")
        ngc_penalty = ngc.penalty(model.causal_gate)

        if config.optimizer == "ista" and train:
            loss = mse + smooth
            logged_loss = loss + ngc_penalty.detach()
        else:
            loss = mse + smooth + ngc_penalty
            logged_loss = loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if config.optimizer == "ista":
                if model.causal_gate is None:
                    raise RuntimeError("ISTA requires use_causal_gate=True")
                ngc.prox_(model.causal_gate, config.learning_rate)
                ngc_penalty = ngc.penalty(model.causal_gate)
                logged_loss = mse.detach() + smooth.detach() + ngc_penalty.detach()

        totals["loss"] += float(logged_loss.detach().cpu())
        totals["mse"] += float(mse.detach().cpu())
        totals["ngc"] += float(ngc_penalty.detach().cpu())
        totals["smooth"] += float(smooth.detach().cpu())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def _infer(
    model: GVARWithNGCGates,
    dataset: LaggedDataset,
    config: GVARTrainingConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    coeffs_all = []
    for start in range(0, dataset.predictors.shape[0], config.batch_size):
        stop = start + config.batch_size
        inputs = torch.as_tensor(dataset.predictors[start:stop], dtype=torch.float32, device=device)
        _, coeffs = model(inputs)
        coeffs_all.append(coeffs.cpu())

    coeffs_t = torch.cat(coeffs_all, dim=0)
    strength_t = model.coefficient_strength(coeffs_t, aggregation=config.strength_aggregation)
    
    if model.causal_gate is not None:
      graph_t = model.causal_graph_from_gate(threshold=config.causal_threshold)
    else:
      graph_t = (strength_t > config.causal_threshold).to(torch.int64)


    return coeffs_t.numpy(), strength_t.numpy(), graph_t.cpu().numpy()


def fit_gvar_ngc(
    data: np.ndarray | list[np.ndarray],
    config: GVARTrainingConfig,
    model: GVARWithNGCGates | None = None,
) -> FitResult:
    """Fit GVAR with NGC-style structured sparsity."""
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    dataset = construct_lagged_dataset(data, order=config.order)
    device = _resolve_device(config.device)

    num_vars = dataset.predictors.shape[-1]
    if model is None:
        model = GVARWithNGCGates(
            num_vars=num_vars,
            order=config.order,
            hidden_layer_size=config.hidden_layer_size,
            num_hidden_layers=config.num_hidden_layers,
            use_causal_gate=True,
            gate_init=config.gate_init,
        )
    model.to(device)

    ngc = NGCRegularizer(name=config.regularizer, lam=config.lambda_ngc, reduction="sum", lag_dim=0)
    optimizer = _make_optimizer(config, model)
    criterion = nn.MSELoss(reduction="mean")
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {"loss": [], "mse": [], "ngc": [], "smooth": []}

    for _ in range(config.max_epochs):
        metrics = _epoch(
            model=model,
            dataset=dataset,
            config=config,
            optimizer=optimizer,
            ngc=ngc,
            criterion=criterion,
            device=device,
            rng=rng,
        )
        for key, value in metrics.items():
            history[key].append(value)

    coeffs, strength, graph = _infer(model, dataset, config, device)
    return FitResult(
        model=model,
        history=history,
        coeffs=coeffs,
        causal_strength=strength,
        causal_graph=graph,
    )
