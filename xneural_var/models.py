from __future__ import annotations

from typing import Literal

import torch
from torch import nn

AggregationName = Literal["max", "mean", "median"]


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int) -> nn.Sequential:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class GVARWithNGCGates(nn.Module):
    """GVAR/SENN model with an NGC-compatible causal gate.

    The model produces a generalized coefficient tensor with shape
    ``[batch, lag, target, source]``. If ``use_causal_gate`` is true, a learned
    gate with shape ``[lag, target, source]`` multiplies those coefficients.
    Proximal updates on the gate can set Granger blocks to exact zero.
    """

    def __init__(
        self,
        num_vars: int,
        order: int,
        hidden_layer_size: int,
        num_hidden_layers: int = 1,
        use_causal_gate: bool = True,
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_vars <= 0:
            raise ValueError("num_vars must be positive")
        if order <= 0:
            raise ValueError("order must be positive")

        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_causal_gate = use_causal_gate

        self.coeff_nets = nn.ModuleList(
            [
                _make_mlp(
                    input_dim=num_vars,
                    hidden_dim=hidden_layer_size,
                    output_dim=num_vars * num_vars,
                    num_hidden_layers=num_hidden_layers,
                )
                for _ in range(order)
            ]
        )

        if use_causal_gate:
            gate = torch.full((order, num_vars, num_vars), float(gate_init))
            self.causal_gate = nn.Parameter(gate)
        else:
            self.register_parameter("causal_gate", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, lag, variables]")
        if inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(
                f"expected inputs shape [batch, {self.order}, {self.num_vars}], "
                f"got {tuple(inputs.shape)}"
            )

        coeffs_by_lag = []
        preds = inputs.new_zeros((inputs.shape[0], self.num_vars))

        for lag_idx, coeff_net in enumerate(self.coeff_nets):
            coeffs_k = coeff_net(inputs[:, lag_idx, :])
            coeffs_k = coeffs_k.reshape(inputs.shape[0], self.num_vars, self.num_vars)
            if self.causal_gate is not None:
                coeffs_k = coeffs_k * self.causal_gate[lag_idx].unsqueeze(0)
            preds = preds + torch.matmul(coeffs_k, inputs[:, lag_idx, :].unsqueeze(-1)).squeeze(-1)
            coeffs_by_lag.append(coeffs_k)

        coeffs = torch.stack(coeffs_by_lag, dim=1)
        return preds, coeffs

    @torch.no_grad()
    def gate_group_norms(self) -> torch.Tensor:
        """Return ``[target, source]`` norms across lag gates."""
        if self.causal_gate is None:
            raise RuntimeError("gate_group_norms requires use_causal_gate=True")
        return torch.linalg.vector_norm(self.causal_gate, ord=2, dim=0)

    @torch.no_grad()
    def causal_graph_from_gate(self, threshold: float = 0.0) -> torch.Tensor:
        """Return binary Granger graph from exact gate groups."""
        return (self.gate_group_norms() > threshold).to(torch.int64)

    @staticmethod
    def coefficient_strength(coeffs: torch.Tensor, aggregation: AggregationName = "max") -> torch.Tensor:
        """Aggregate coefficients into a ``[target, source]`` causal strength map."""
        abs_coeffs = coeffs.abs()
        if aggregation == "max":
            return abs_coeffs.amax(dim=(0, 1))
        if aggregation == "mean":
            return abs_coeffs.mean(dim=(0, 1))
        if aggregation == "median":
            return abs_coeffs.median(dim=0).values.median(dim=0).values
        raise ValueError(f"unsupported aggregation: {aggregation}")
