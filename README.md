# XNeural VAR

Modular PyTorch implementation of a GVAR-style self-explaining neural
autoregression with Neural Granger Causality (NGC) regularization.

This code is designed around two reference implementations:

- GVAR: <https://github.com/i6092467/GVAR>
- Neural-GC: <https://github.com/iancovert/Neural-GC>

The main adaptation is a learnable `causal_gate` with shape
`[lag, target, source]`. GVAR coefficient networks produce time-varying
generalized coefficient matrices, and every coefficient is multiplied by this
gate. NGC-style sparse group lasso is then applied to the gate so proximal
gradient updates can create exact zeros in the learned Granger structure.

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
    regularizer="sparse_group_lasso",
    optimizer="ista",
    log_every=10,
    verbose=1,
)

result = fit_gvar_ngc(data, config)

print(result.causal_strength)
print(result.causal_graph)
```

Example training output:

```text
Epoch    1/200  | loss=1.72 | mse=1.51 | ngc=0.21 | smooth=0 | active_edges=25/25 (100.00%)
Epoch   10/200  | loss=1.38 | mse=1.22 | ngc=0.16 | smooth=0 | active_edges=18/25 (72.00%)
```

## Sparse Group Lasso

Neural-GC's cMLP implementation supports a `GSGL` penalty, described in the
repository as group sparse group lasso. Its proximal update first applies a
lag-level shrinkage and then applies source-level group shrinkage.

In this package the regularized object is not an input-layer tensor but
`causal_gate`:

```python
causal_gate.shape == [lag, target, source]
```

Because each lag-level gate entry is scalar, the Neural-GC lag-level group
shrinkage becomes elementwise L1 shrinkage. The edge-level term remains group
lasso over the whole lag vector for each target-source pair.

For each edge `source j -> target i`, define:

```math
g_{i,j} = (G_{1,i,j}, \dots, G_{K,i,j})
```

The default NGC regularizer is:

```math
R_{\mathrm{SGL}}(G)
=
\lambda_{\mathrm{ngc}}
\left(
  w_{\mathrm{group}}
  \sum_{i,j} \|g_{i,j}\|_2
  +
  w_{\mathrm{l1}}
  \sum_{k,i,j} |G_{k,i,j}|
\right)
```

`sparse_group_weight` and `sparse_l1_weight` control the two weights. Both
default to `1.0`, matching the Neural-GC `GSGL` update style.

With `optimizer="ista"`, the smooth prediction objective is optimized first,
then the sparse-group proximal operator is applied to `causal_gate`:

1. elementwise soft-thresholding for lag-level sparsity;
2. group soft-thresholding over all lags for each target-source edge.

This can remove individual lag gates and can also remove the entire Granger
edge.

## Design Notes

- The regularization target is `causal_gate`, not input or output layer
  weights. In GVAR, Granger structure is read from generalized coefficients,
  so sparsifying a gate on those coefficients gives a direct `source -> target`
  zero pattern.
- Coefficient-network weight decay is separated from the gate. This mitigates
  the scale non-identifiability of `raw_coefficient * causal_gate` without
  weakening exact-zero proximal updates.
- `hierarchical_group_lasso` is still available as an optional structured
  penalty. Lag index `0` is the most distant lag, so nested hierarchical groups
  are prefixes: `[t - order]`, `[t - order, t - order + 1]`, ...,
  `[t - order, ..., t - 1]`.
- `optimizer="adam"` adds the nonsmooth penalty directly to the objective. This
  is easy to optimize but does not guarantee exact zeros.
- `optimizer="ista"` optimizes the smooth objective and then applies the
  proximal operator to `causal_gate`, producing exact zeros in the learned
  causal structure.
- The default causal graph threshold is `1e-8` to avoid treating tiny numerical
  residue as a nonzero edge.

## Training Logs

Training logs are controlled by `verbose` and `log_every`.

- `verbose=0`: no logs.
- `verbose=1`: print epoch, loss, MSE, NGC penalty, smoothness penalty, and
  active edge count.
- `log_every=10`: print every 10 epochs, plus the first and final epoch.

## Package Layout

- `xneural_var.models`: GVAR/SENN model with causal gates.
- `xneural_var.regularizers`: sparse group lasso, hierarchical group lasso, and
  proximal operators.
- `xneural_var.training`: training loop, logging, and result objects.
- `xneural_var.data`: lagged dataset construction.
