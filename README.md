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
    coefficient_weight_decay=1e-4,
    regularizer="sparse_group_lasso",
    sparse_group_lambda=1e-2,
    sparse_l1_lambda=1e-3,
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
\lambda_{\mathrm{group}}
\sum_{i,j} \|g_{i,j}\|_2
+
\lambda_{\mathrm{l1}}
\sum_{k,i,j} |G_{k,i,j}|
```

`sparse_group_lambda` and `sparse_l1_lambda` control the two penalty strengths
directly. `lambda_ngc` is not used by `sparse_group_lasso`; it is reserved for
`group_lasso` and `hierarchical_group_lasso`.

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

## Baselines

For validation experiments, the package also exposes reference-style cMLP and
GVAR baselines with the same high-level fit-result interface:

```python
from xneural_var import CMLPTrainingConfig, GVARBaselineTrainingConfig
from xneural_var import fit_cmlp, fit_gvar

cmlp_result = fit_cmlp(
    data,
    CMLPTrainingConfig(
        order=4,
        hidden_layer_size=32,
        regularizer="sparse_group_lasso",
        sparse_group_lambda=1e-2,
        sparse_l1_lambda=1e-3,
        optimizer="ista",
    ),
)

gvar_result = fit_gvar(
    data,
    GVARBaselineTrainingConfig(
        order=4,
        hidden_layer_size=32,
        lambda_coeff=1e-3,
        lambda_smooth=1e-3,
    ),
)
```

`fit_cmlp` follows the Neural-GC component-wise MLP design: one MLP is trained
per target variable, and Granger structure is read from the first-layer input
weights. Its sparse group lasso uses direct `sparse_group_lambda` and
`sparse_l1_lambda` parameters, matching the XNeural VAR convention.

`fit_gvar` disables the NGC causal gate and trains the GVAR/SENN-style
state-dependent coefficient model directly. Because this baseline has no
proximal gate, its reported graph is thresholded from coefficient strength
rather than exact structural zeros.

### About the Comparison Implementations

`fit_cmlp` follows the official Neural-GC implementation and trains an independent component-wise MLP for each target variable. The Granger structure is read from the input weights of the first layer.

`fit_gvar` is a pure GVAR/SENN-style model that does not use causal gates. Since this model does not produce exact zeros through proximal gradient updates, `causal_graph` should be interpreted as a comparison graph obtained by thresholding coefficient magnitudes with `causal_threshold`.

## Package Layout

- `xneural_var.models`: GVAR/SENN model with causal gates.
- `xneural_var.gvar`: GVAR baseline without NGC causal gates.
- `xneural_var.cmlp`: Neural-GC cMLP baseline.
- `xneural_var.regularizers`: sparse group lasso, hierarchical group lasso, and
  proximal operators.
- `xneural_var.training`: training loop, logging, and result objects.
- `xneural_var.data`: lagged dataset construction.

---

## アルゴリズム: self-eXplaining neural VAR

本実装は、Generalized Vector Autoregression (GVAR) に Neural Granger
Causality (NGC) 型の構造的スパース性を導入したモデルである。

目的は、非線形かつ状態依存的な時系列ダイナミクスを表現しつつ、
Granger 非因果性に対応する係数ブロックを近接勾配法により厳密に
ゼロ化することである。今回の標準設定では、NGC 公式実装の `GSGL`
に対応する sparse group lasso を causal gate に適用する。

### 1. モデル

`p` 変量時系列を次のように表す。

```math
x_t \in \mathbb{R}^p
```

自己回帰次数を `K` とし、時点 `t` におけるラグ付き入力を次のように
定義する。

```math
X_t = (x_{t-K}, \dots, x_{t-1})
```

各ラグ `k = 1, ..., K` に対して、GVAR は状態依存係数行列を
ニューラルネットワークにより生成する。

```math
\Phi_k(x_{t-k}) \in \mathbb{R}^{p \times p}
```

一歩先予測は次で定義される。

```math
\hat{x}_t
=
\sum_{k=1}^{K}
\tilde{\Phi}_k(x_{t-k}) x_{t-k}
```

ここで、構造的な Granger sparsity を表す static causal gate を導入する。

```math
G_k \in \mathbb{R}^{p \times p}
```

有効係数行列を次で定義する。

```math
\tilde{\Phi}_k(x_{t-k})
=
G_k \odot \Phi_k(x_{t-k})
```

ここで、`\odot` は Hadamard 積である。実装上、係数テンソルは次の形を
持つ。

```python
[batch, lag, target, source]
```

causal gate は次の形を持つ。

```python
[lag, target, source]
```

### 2. Granger 非因果性

変数 `x_j` が変数 `x_i` を Granger cause しないとは、すべてのラグに
おいて `x_j` から `x_i` への効果がゼロであることを意味する。

本実装では、この条件を causal gate により次のように表現する。

```math
G_{1,i,j}
=
G_{2,i,j}
=
\cdots
=
G_{K,i,j}
=
0
```

したがって、推定される Granger causal graph は次で定義される。

```math
\hat{A}_{i,j}
=
\mathbf{1}
\left[
\left\|
(G_{1,i,j}, \dots, G_{K,i,j})
\right\|_2
>
\tau
\right]
```

ここで、`\tau` は数値誤差対策のための小さな閾値である。現在の
デフォルトは `1e-8` である。

### 3. 目的関数

学習目的関数は次である。

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
+
\mathcal{R}_{\mathrm{ngc}}
```

ここで、`sparse_group_lasso` の場合は `\mathcal{R}_{\mathrm{ngc}}`
自体が `sparse_group_lambda` と `sparse_l1_lambda` を含む。一方、
`hierarchical_group_lasso` や legacy な `group_lasso` では、従来通り
`lambda_ngc` が正則化全体の強さを制御する。

ただし、optimizer として `ista` を用いる場合、NGC 正則化項は通常の
勾配ステップには含めない。その代わり、causal gate に対して
近接作用素を直接適用する。

### 4. 予測損失

予測損失は平均二乗誤差である。

```math
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{N}
\sum_t
\left\|
x_t - \hat{x}_t
\right\|_2^2
```

実装では次を用いる。

```python
nn.MSELoss(reduction="mean")
```

### 5. 時間方向の平滑化ペナルティ

GVAR は時点ごとに状態依存係数を生成するため、係数が過度に振動しない
ように平滑化ペナルティを導入する。

時点 `t` における有効係数テンソルを `\tilde{\Phi}_t` と表す。
absolute mode では、平滑化ペナルティは次である。

```math
\mathcal{R}_{\mathrm{smooth}}
=
\frac{1}{|\mathcal{T}|}
\sum_{t \in \mathcal{T}}
\left\|
\tilde{\Phi}_{t+1}
-
\tilde{\Phi}_t
\right\|_F^2
```

relative mode では、係数スケールで正規化した次のペナルティを用いる。

```math
\mathcal{R}_{\mathrm{smooth}}
=
\frac{1}{|\mathcal{T}|}
\sum_{t \in \mathcal{T}}
\frac{
\left\|
\tilde{\Phi}_{t+1}
-
\tilde{\Phi}_t
\right\|_F^2
}{
\left\|
\tilde{\Phi}_t
\right\|_F^2
+
\varepsilon
}
```

実装では、`time_index` を用いて次を満たす隣接時点のみに平滑化
ペナルティを課す。

```math
t_{r+1} - t_r = 1
```

これにより、複数系列や複数 replicate がある場合でも、系列境界を
またいだ smoothing を避ける。

### 6. NGC 正則化

#### 6.1 Sparse Group Lasso

現在の標準設定では、NGC 公式実装の `GSGL` を参考に sparse group lasso
を用いる。target-source pair `(i, j)` ごとに、ラグ方向の gate vector
を次のように定義する。

```math
g_{i,j}
=
(G_{1,i,j}, \dots, G_{K,i,j})
\in \mathbb{R}^{K}
```

sparse group lasso penalty は次である。

```math
\mathcal{R}_{\mathrm{ngc}}
=
\lambda_{\mathrm{group}}
\sum_{i=1}^{p}
\sum_{j=1}^{p}
\left\|
g_{i,j}
\right\|_2
+
\lambda_{\mathrm{l1}}
\sum_{k=1}^{K}
\sum_{i=1}^{p}
\sum_{j=1}^{p}
\left|
G_{k,i,j}
\right|
```

実装上は、これらを次の設定値で直接指定する。

```python
sparse_group_lambda = 1e-2
sparse_l1_lambda = 1e-3
```

第1項は edge 単位の group sparsity を作り、ある pair `(i, j)` の全ラグを
同時にゼロ化する方向に働く。これは Granger 非因果性に直接対応する。

```math
x_j \not\to x_i
```

第2項は lag 単位の elementwise sparsity を作る。これにより、edge 全体は
残しつつ、一部のラグだけをゼロ化することもできる。

NGC 公式の cMLP では、入力層重みに hidden 次元があるため `GSGL` の
最初の shrinkage は lag-level group shrinkage になる。本実装の
`causal_gate` は各 lag の gate が scalar なので、その対応物は L1
soft-thresholding になる。

#### 6.2 Hierarchical Group Lasso

optional に hierarchical group lasso も利用できる。

現在の実装では、lag index `0` が最も古いラグに対応すると仮定している。
この仮定の下で、nested prefix group を次のように定義する。

```math
g_{i,j}^{(k)}
=
(G_{1,i,j}, \dots, G_{k,i,j})
```

hierarchical penalty は次である。

```math
\mathcal{R}_{\mathrm{hier}}
=
\sum_{k=1}^{K}
\sum_{i=1}^{p}
\sum_{j=1}^{p}
\left\|
g_{i,j}^{(k)}
\right\|_2
```

この解釈が成立するには、データセットの lag ordering が次を満たす必要が
ある。

```python
inputs[:, 0, :] = x_{t-K}
inputs[:, K-1, :] = x_{t-1}
```

もし逆順の lag ordering を用いる場合、hierarchical penalty の方向を
反転する必要がある。

### 7. 最適化

本実装では、optimizer として次の2つを選択できる。

```python
optimizer = "adam"
```

または

```python
optimizer = "ista"
```

`adam` を用いる場合、目的関数全体を通常の勾配法で最適化する。ただし、
Adam では causal gate が厳密にゼロになりにくい。exact sparsity を
重視する場合は `ista` を用いる。

`ista` を用いる場合、まず smooth part のみに対して勾配ステップを行う。

```math
\mathcal{L}_{\mathrm{smooth\text{-}part}}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
```

勾配ステップは次である。

```math
\theta^{(m+1/2)}
=
\theta^{(m)}
-
\eta
\nabla_{\theta}
\mathcal{L}_{\mathrm{smooth\text{-}part}}
```

その後、causal gate `G` に対して sparse group lasso の近接作用素を
適用する。

```math
G^{(m+1)}
=
\mathrm{prox}_{
\eta \mathcal{R}_{\mathrm{ngc}}
}
\left(
G^{(m+1/2)}
\right)
```

### 8. Sparse Group Lasso の近接作用素

本実装では、NGC 公式の `GSGL` と同じ順序で近接更新を行う。

まず elementwise soft-thresholding により lag 単位の sparsity を作る。

```math
G_{k,i,j}
\leftarrow
\mathrm{sign}(G_{k,i,j})
\left(
\left|G_{k,i,j}\right|
-
\eta \lambda_{\mathrm{l1}}
\right)_+
```

次に、各 pair `(i, j)` の lag vector `g_{i,j}` に対して group
soft-thresholding を行う。

```math
g_{i,j}
\leftarrow
\left(
1
-
\frac{
\eta \lambda_{\mathrm{group}}
}{
\left\|
g_{i,j}
\right\|_2
}
\right)_+
g_{i,j}
```

この2段階の近接更新により、個別の lag gate と Granger edge 全体の
両方に厳密なゼロが生じる。

### 9. Graph Inference

学習後、モデルは全データに対して有効係数テンソルを計算する。

係数ベースの causal strength は、係数の絶対値を batch 方向および lag
方向に集約して得る。max aggregation の場合、causal strength は次である。

```math
S_{i,j}
=
\max_{t,k}
\left|
\tilde{\Phi}_{k,i,j}(x_{t-k})
\right|
```

ただし、causal gate が存在する場合、最終的な causal graph は coefficient
thresholding ではなく gate norm から推定する。

```math
\hat{A}_{i,j}
=
\mathbf{1}
\left[
\left\|
g_{i,j}
\right\|_2
>
\tau
\right]
```

### 10. Parameter Groups

optimizer では、パラメータを次の2グループに分ける。

```python
coefficient_network_parameters
causal_gate_parameters
```

coefficient-generating neural networks には optional に weight decay を
適用できる。

```math
\lambda_{\mathrm{wd}}
\left\|
\theta_{\Phi}
\right\|_2^2
```

一方で、causal gate には weight decay を適用しない。causal gate の
sparsity は通常の weight decay ではなく、近接作用素により制御する。

### 11. 学習ログ

`verbose` と `log_every` により学習ログを表示できる。

```python
verbose = 1
log_every = 10
```

表示される主な値は次の通りである。

- `loss`: 予測損失、smoothness、NGC 正則化を含むログ用損失
- `mse`: 予測損失
- `ngc`: causal gate に対する sparse group lasso penalty
- `smooth`: 時間方向の平滑化 penalty
- `active_edges`: gate norm が閾値を超える Granger edge の数

### 12. まとめ

本アルゴリズムの特徴は次の通りである。

1. GVAR により、非線形かつ状態依存的な自己回帰ダイナミクスを表現する。
2. static causal gate により、構造的 Granger sparsity と動的係数変動を分離する。
3. sparse group lasso により、edge 全体の削除と lag 単位の削除を同時に扱う。
4. proximal gradient update により、causal gate に厳密なゼロを生成する。
5. gate group が厳密にゼロであることは Granger 非因果性に対応する。
6. temporal smoothness regularization により、状態依存係数の過度な時間変動を抑制する。
7. causal graph は coefficient thresholding ではなく gate norm に基づいて推定する。
