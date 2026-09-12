from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .models import GVARWithNGCGates


@dataclass(frozen=True)
class JacobianAgreementResult:
    """Coefficient/Jacobian diagnostics for lagged model inputs.

    Array fields use ``[sample, lag, target, source]``.  ``active_*`` metrics
    exclude structurally zero gate entries so exact zeros cannot make the
    reported agreement look artificially strong.
    """

    coefficients: np.ndarray
    jacobian: np.ndarray
    mismatch: np.ndarray
    mse: float
    mae: float
    relative_frobenius_error: float
    cosine_similarity: float
    active_mse: float
    active_mae: float
    active_sign_agreement: float
    active_count: int


def _model_device_and_dtype(
    model: GVARWithNGCGates,
) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def _cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
    eps: float,
) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= eps and right_norm <= eps:
        return 1.0
    if left_norm <= eps or right_norm <= eps:
        return 0.0
    similarity = np.dot(left, right) / (left_norm * right_norm)
    return float(np.clip(similarity, -1.0, 1.0))


def evaluate_jacobian_agreement(
    model: GVARWithNGCGates,
    inputs: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 256,
    active_threshold: float = 1e-8,
    zero_tolerance: float = 1e-8,
) -> JacobianAgreementResult:
    """Evaluate local coefficient/Jacobian agreement on lagged predictors.

    Parameters
    ----------
    model:
        A fitted gated XNeural-VAR model.
    inputs:
        Lagged predictors with shape ``[sample, lag, variables]``.  Raw time
        series can be converted with :func:`construct_lagged_dataset`.
    batch_size:
        Number of samples used for each Jacobian calculation.
    active_threshold:
        A gate entry is included in ``active_*`` metrics when its absolute
        value is greater than this threshold.
    zero_tolerance:
        Magnitudes at or below this value count as zero for sign agreement.
    """
    if model.causal_gate is None:
        raise ValueError(
            "evaluate_jacobian_agreement requires use_causal_gate=True"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(active_threshold) or active_threshold < 0:
        raise ValueError("active_threshold must be non-negative")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")

    device, dtype = _model_device_and_dtype(model)
    inputs_t = torch.as_tensor(inputs, dtype=dtype, device=device)
    if inputs_t.ndim != 3:
        raise ValueError("inputs must have shape [sample, lag, variables]")
    if inputs_t.shape[0] == 0:
        raise ValueError("inputs must contain at least one sample")

    was_training = model.training
    model.eval()
    coefficients = []
    jacobians = []
    mismatches = []
    try:
        for start in range(0, inputs_t.shape[0], batch_size):
            batch = inputs_t[start : start + batch_size]
            _, coeffs, jacobian, mismatch = model.forward_with_jacobian(batch)
            coefficients.append(coeffs.detach().cpu())
            jacobians.append(jacobian.detach().cpu())
            mismatches.append(mismatch.detach().cpu())
    finally:
        model.train(was_training)

    coefficients_np = torch.cat(coefficients).numpy()
    jacobian_np = torch.cat(jacobians).numpy()
    mismatch_np = torch.cat(mismatches).numpy()

    mse = float(np.mean(np.square(mismatch_np)))
    mae = float(np.mean(np.abs(mismatch_np)))
    jacobian_flat = jacobian_np.reshape(-1).astype(np.float64, copy=False)
    coefficients_flat = coefficients_np.reshape(-1).astype(
        np.float64,
        copy=False,
    )
    mismatch_flat = mismatch_np.reshape(-1).astype(np.float64, copy=False)
    metric_eps = max(zero_tolerance, np.finfo(np.float64).eps)
    denominator = max(float(np.linalg.norm(jacobian_flat)), metric_eps)
    relative_error = float(np.linalg.norm(mismatch_flat) / denominator)
    cosine = _cosine_similarity(
        jacobian_flat,
        coefficients_flat,
        metric_eps,
    )

    active_gate = (
        model.causal_gate.detach().abs().cpu().numpy() > active_threshold
    )
    active_values = np.broadcast_to(
        active_gate,
        mismatch_np.shape,
    )
    active_count = int(active_values.sum())
    if active_count:
        active_mismatch = mismatch_np[active_values]
        active_mse = float(np.mean(np.square(active_mismatch)))
        active_mae = float(np.mean(np.abs(active_mismatch)))

        active_jacobian = jacobian_np[active_values]
        active_coefficients = coefficients_np[active_values]
        jacobian_sign = np.sign(
            np.where(
                np.abs(active_jacobian) <= zero_tolerance,
                0.0,
                active_jacobian,
            )
        )
        coefficient_sign = np.sign(
            np.where(
                np.abs(active_coefficients) <= zero_tolerance,
                0.0,
                active_coefficients,
            )
        )
        active_sign_agreement = float(
            np.mean(jacobian_sign == coefficient_sign)
        )
    else:
        active_mse = float("nan")
        active_mae = float("nan")
        active_sign_agreement = float("nan")

    return JacobianAgreementResult(
        coefficients=coefficients_np,
        jacobian=jacobian_np,
        mismatch=mismatch_np,
        mse=mse,
        mae=mae,
        relative_frobenius_error=relative_error,
        cosine_similarity=cosine,
        active_mse=active_mse,
        active_mae=active_mae,
        active_sign_agreement=active_sign_agreement,
        active_count=active_count,
    )
