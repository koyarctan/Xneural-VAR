from __future__ import annotations

from typing import Literal

import torch
from torch import nn

AggregationName = Literal["max", "mean", "median"]


class LagwiseMLP(nn.Module):
    """Independent MLP per lag, evaluated as batched tensor operations."""

    def __init__(
        self,
        num_lags: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        self.num_lags = num_lags
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(num_lags, dims[layer_idx + 1], dims[layer_idx]))
                for layer_idx in range(len(dims) - 1)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(num_lags, dims[layer_idx + 1]))
                for layer_idx in range(len(dims) - 1)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weights, self.biases):
            for lag_idx in range(self.num_lags):
                nn.init.xavier_normal_(weight[lag_idx])
            nn.init.constant_(bias, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = inputs
        final_layer = len(self.weights) - 1
        for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = torch.einsum("bki,koi->bko", activations, weight) + bias.unsqueeze(0)
            if layer_idx != final_layer:
                activations = torch.relu(activations)
        return activations


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

        self.coeff_net = LagwiseMLP(
            num_lags=order,
            input_dim=num_vars,
            hidden_dim=hidden_layer_size,
            output_dim=num_vars * num_vars,
            num_hidden_layers=num_hidden_layers,
        )

        if use_causal_gate:
            gate = torch.full((order, num_vars, num_vars), float(gate_init))
            self.causal_gate = nn.Parameter(gate)
        else:
            self.register_parameter("causal_gate", None)

    def reset_parameters(self) -> None:
        self.coeff_net.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, lag, variables]")
        if inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(
                f"expected inputs shape [batch, {self.order}, {self.num_vars}], "
                f"got {tuple(inputs.shape)}"
            )

        coeffs = self.coeff_net(inputs).reshape(inputs.shape[0], self.order, self.num_vars, self.num_vars)
        if self.causal_gate is not None:
            coeffs = coeffs * self.causal_gate.unsqueeze(0)
        preds = torch.einsum("bkij,bkj->bi", coeffs, inputs)
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
