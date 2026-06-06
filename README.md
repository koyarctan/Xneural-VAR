# XNeural VAR

Modular PyTorch implementation of a GVAR-style self-explaining neural
autoregression with Neural Granger Causality (NGC) regularization.

This code is designed around two reference implementations:

- GVAR: <https://github.com/i6092467/GVAR>
- Neural-GC: <https://github.com/iancovert/Neural-GC>

The main adaptation is a learnable `causal_gate` with shape
`[lag, target, source]`. The GVAR coefficient networks still produce
time-varying generalized coefficient matrices, but every coefficient is
multiplied by the gate. Group lasso or hierarchical group lasso can then be
applied to the gate, and a proximal step can set whole causal blocks to exact
zero.

## Installation

```bash
pip install -e .
```

## Minimal Usage

```python
import numpy as np

from xneural_var import GVARTrainingConfig, fit_gvar_ngc

data = np.random.randn(300, 5).astype("float32")

config = GVARTrainingConfig(
    order=4,
    hidden_layer_size=32,
    num_hidden_layers=2,
    max_epochs=200,
    batch_size=64,
    learning_rate=1e-3,
    lambda_ngc=1e-2,
    coefficient_weight_decay=1e-4,
    regularizer="hierarchical_group_lasso",
    optimizer="ista",
)

result = fit_gvar_ngc(data, config)

print(result.causal_strength)
print(result.causal_graph)
```

## Design Notes

- The default regularization target is `causal_gate`, not the input or output
  layer weights. This is deliberate: in GVAR, Granger structure is read from
  generalized coefficients, so sparsifying a gate on those coefficients gives a
  direct `source -> target` zero pattern.
- Coefficient-network weight decay is separated from the gate. This mitigates
  the scale non-identifiability of `raw_coefficient * causal_gate` without
  weakening the exact-zero proximal update on the gate.
- `group_lasso` groups all lags for each target-source pair.
- `hierarchical_group_lasso` uses nested lag-prefix groups, following the
  Neural-GC cMLP implementation convention where lag index `0` is the most
  distant lag. With this package's lag layout, the nested groups are prefixes:
  `[t - order]`, `[t - order, t - order + 1]`, ..., `[t - order, ..., t - 1]`.
- `optimizer="adam"` adds the nonsmooth penalty directly to the objective.
  This is easy to optimize but does not guarantee exact zero coefficients.
- `optimizer="ista"` optimizes the smooth objective and then applies the
  proximal operator to `causal_gate`, producing exact zeros in the learned
  causal structure.
- GVAR temporal smoothness is implemented as an optional adjacent-time penalty
  over inferred generalized coefficients. Use `smoothness_mode="relative"` for
  scale-normalized smoothing.
- The default causal graph threshold is `1e-8` to avoid treating tiny numerical
  residue as a nonzero edge.

## Package Layout

- `xneural_var.models`: GVAR/SENN model with optional causal gates.
- `xneural_var.regularizers`: group lasso, hierarchical group lasso, and
  proximal operators.
- `xneural_var.training`: training loop and result objects.
- `xneural_var.data`: lagged dataset construction.
