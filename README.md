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

---

## アルゴリズム: GVAR with NGC-style Proximal Group Sparsity

本実装は、Generalized Vector Autoregression (GVAR) に Neural Granger Causality (NGC) 型の構造的スパース性を導入したモデルである。

目的は、非線形かつ状態依存的な時系列ダイナミクスを表現しつつ、Granger 非因果性に対応する係数ブロックを近接勾配法により厳密にゼロ化することである。

---

## 1. モデル

\(p\) 変量時系列を

$$
x_t \in \mathbb{R}^p
$$

とする。自己回帰次数を \(K\) とし、時点 \(t\) におけるラグ付き入力を

$$
X_t = (x_{t-K}, \dots, x_{t-1})
$$

とする。

各ラグ \(k \in \{1,\dots,K\}\) に対して、GVAR は状態依存係数行列

$$
\Phi_k(x_{t-k}) \in \mathbb{R}^{p \times p}
$$

をニューラルネットワークにより生成する。

一歩先予測は次で定義される。

$$
\hat{x}_t
=
\sum_{k=1}^{K}
\tilde{\Phi}_k(x_{t-k}) x_{t-k}
$$

ここで、構造的な Granger sparsity を表す static causal gate

$$
G_k \in \mathbb{R}^{p \times p}
$$

を導入する。

有効係数行列を

$$
\tilde{\Phi}_k(x_{t-k})
=
G_k \odot \Phi_k(x_{t-k})
$$

と定義する。

ここで、\(\odot\) は Hadamard 積である。

実装上、係数テンソルは次の形を持つ。

```python
[batch, lag, target, source]
```

causal gate は次の形を持つ。

```python
[lag, target, source]
```

---

## 2. Granger 非因果性

変数 \(x_j\) が変数 \(x_i\) を Granger cause しないとは、すべてのラグにおいて \(x_j\) から \(x_i\) への効果がゼロであることを意味する。

本実装では、この条件を causal gate により次のように表現する。

$$
G_{1,i,j}
=
G_{2,i,j}
=
\cdots
=
G_{K,i,j}
=
0
$$

したがって、推定される Granger causal graph は

$$
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
$$

で定義される。

ここで、\(\tau\) は数値誤差対策のための小さな閾値である。

この定義により、edge \(j \to i\) はラグ全体を1つのグループとして選択または削除される。

---

## 3. 目的関数

学習目的関数は次である。

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
+
\lambda_{\mathrm{ngc}}
\mathcal{R}_{\mathrm{ngc}}
$$

ただし、optimizer として `ista` を用いる場合、NGC 正則化項は通常の勾配ステップには含めない。

その代わり、causal gate に対して近接作用素を直接適用する。

---

## 4. 予測損失

予測損失は平均二乗誤差である。

$$
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{N}
\sum_t
\left\|
x_t - \hat{x}_t
\right\|_2^2
$$

実装では次を用いる。

```python
nn.MSELoss(reduction="mean")
```

---

## 5. 時間方向の平滑化ペナルティ

GVAR は時点ごとに状態依存係数を生成するため、係数が過度に振動しないように平滑化ペナルティを導入する。

時点 \(t\) における有効係数テンソルを \(\tilde{\Phi}_t\) と表す。

absolute mode では、平滑化ペナルティは次である。

$$
\mathcal{R}_{\mathrm{smooth}}
=
\frac{1}{|\mathcal{T}|}
\sum_{t \in \mathcal{T}}
\left\|
\tilde{\Phi}_{t+1}
-
\tilde{\Phi}_t
\right\|_F^2
$$

relative mode では、係数スケールで正規化した次のペナルティを用いる。

$$
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
$$

実装では、`time_index` を用いて

$$
t_{r+1} - t_r = 1
$$

を満たす隣接時点のみに平滑化ペナルティを課す。

これにより、複数系列や複数 replicate がある場合でも、系列境界をまたいだ smoothing を避ける。

---

## 6. NGC 正則化

### 6.1 Group Lasso

標準設定では、target-source pair \((i,j)\) ごとに、ラグ方向の gate vector

$$
g_{i,j}
=
(G_{1,i,j}, \dots, G_{K,i,j})
\in \mathbb{R}^{K}
$$

を1つのグループとして扱う。

group lasso penalty は

$$
\mathcal{R}_{\mathrm{ngc}}
=
\sum_{i=1}^{p}
\sum_{j=1}^{p}
\left\|
g_{i,j}
\right\|_2
$$

である。

このペナルティは、ある pair \((i,j)\) に対して、すべてのラグの gate を同時にゼロ化する方向に働く。

したがって、これは Granger 非因果性

$$
x_j \not\to x_i
$$

に直接対応する。

---

### 6.2 Hierarchical Group Lasso

optional に hierarchical group lasso も利用できる。

現在の実装では、lag index \(0\) が最も古いラグに対応すると仮定している。

この仮定の下で、nested prefix group を

$$
g_{i,j}^{(k)}
=
(G_{1,i,j}, \dots, G_{k,i,j})
$$

と定義する。

hierarchical penalty は次である。

$$
\mathcal{R}_{\mathrm{hier}}
=
\sum_{k=1}^{K}
\sum_{i=1}^{p}
\sum_{j=1}^{p}
\left\|
g_{i,j}^{(k)}
\right\|_2
$$

古いラグほど多くの nested group に含まれるため、古いラグに対してより強い shrinkage がかかる。

この解釈が成立するには、データセットの lag ordering が次を満たす必要がある。

```python
inputs[:, 0, :]   = x_{t-K}
inputs[:, K-1, :] = x_{t-1}
```

もし逆順の lag ordering を用いる場合、hierarchical penalty の方向を反転する必要がある。

---

## 7. 最適化

本実装では、optimizer として次の2つを選択できる。

```python
optimizer = "adam"
```

または

```python
optimizer = "ista"
```

---

### 7.1 Adam

`adam` を用いる場合、次の目的関数全体を通常の勾配法で最適化する。

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
+
\lambda_{\mathrm{ngc}}
\mathcal{R}_{\mathrm{ngc}}
$$

ただし、Adam では causal gate が厳密にゼロになりにくい。

したがって、exact sparsity を重視する場合は `ista` を用いる。

---

### 7.2 ISTA / Proximal Gradient

`ista` を用いる場合、まず smooth part のみに対して勾配ステップを行う。

$$
\mathcal{L}_{\mathrm{smooth\mbox{-}part}}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
$$

勾配ステップは次である。

$$
\theta^{(m+1/2)}
=
\theta^{(m)}
-
\eta
\nabla_{\theta}
\mathcal{L}_{\mathrm{smooth\mbox{-}part}}
$$

その後、causal gate \(G\) に対して近接作用素を適用する。

$$
G^{(m+1)}
=
\operatorname{prox}_{
\eta \lambda_{\mathrm{ngc}} \mathcal{R}_{\mathrm{ngc}}
}
\left(
G^{(m+1/2)}
\right)
$$

この近接更新により、causal gate に厳密なゼロが生じる。

---

## 8. Group Lasso の近接作用素

各 pair \((i,j)\) に対して

$$
g_{i,j}
=
(G_{1,i,j}, \dots, G_{K,i,j})
$$

とする。

group lasso の近接作用素は group soft-thresholding であり、次で与えられる。

$$
g_{i,j}
\leftarrow
\left(
1
-
\frac{
\eta \lambda_{\mathrm{ngc}}
}{
\left\|
g_{i,j}
\right\|_2
}
\right)_+
g_{i,j}
$$

ここで、

$$
(a)_+ = \max(a,0)
$$

である。

もし

$$
\left\|
g_{i,j}
\right\|_2
\le
\eta \lambda_{\mathrm{ngc}}
$$

ならば、

$$
g_{i,j} = 0
$$

となる。

したがって、edge \(j \to i\) は厳密に削除される。

---

## 9. Graph Inference

学習後、モデルは全データに対して有効係数テンソルを計算する。

係数ベースの causal strength は、係数の絶対値を batch 方向および lag 方向に集約して得る。

max aggregation の場合、

$$
S_{i,j}
=
\max_{t,k}
\left|
\tilde{\Phi}_{k,i,j}(x_{t-k})
\right|
$$

である。

ただし、causal gate が存在する場合、最終的な causal graph は coefficient thresholding ではなく gate norm から推定する。

$$
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
$$

causal gate が存在しない場合のみ、coefficient strength を thresholding する。

$$
\hat{A}_{i,j}
=
\mathbf{1}
[
S_{i,j} > \tau
]
$$

---

## 10. Parameter Groups

optimizer では、パラメータを次の2グループに分ける。

```python
coefficient_network_parameters
causal_gate_parameters
```

coefficient-generating neural networks には optional に weight decay を適用できる。

$$
\lambda_{\mathrm{wd}}
\left\|
\theta_{\Phi}
\right\|_2^2
$$

一方で、causal gate には weight decay を適用しない。

causal gate の sparsity は通常の weight decay ではなく、近接作用素により制御する。

---

## 11. まとめ

本アルゴリズムの特徴は次の通りである。

1. GVAR により、非線形かつ状態依存的な自己回帰ダイナミクスを表現する。
2. static causal gate により、構造的 Granger sparsity と動的係数変動を分離する。
3. group lasso により、1つの target-source pair の全ラグを1つの Granger edge として扱う。
4. proximal gradient update により、causal gate に厳密なゼロを生成する。
5. gate group が厳密にゼロであることは Granger 非因果性に対応する。
6. temporal smoothness regularization により、状態依存係数の過度な時間変動を抑制する。
7. causal gate が存在する場合、最終的な causal graph は gate norm に基づいて推定する。