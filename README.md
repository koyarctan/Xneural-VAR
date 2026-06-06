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

## Algorithm: GVAR with NGC-style Proximal Group Sparsity

本実装は、Generalized Vector Autoregression (GVAR) に Neural Granger Causality (NGC) 型の構造的スパース性を導入したモデルである。目的は、非線形・状態依存的な時系列力学を表現しつつ、Granger 非因果性に対応する係数ブロックを近接勾配法により厳密にゼロ化することである。

---

### 1. Model

$p$ 変量時系列を

\[
x_t = (x_{t,1}, \dots, x_{t,p})^\top \in \mathbb{R}^p
\]

とする。ラグ次数を $K$ とし、入力は

\[
X_t = (x_{t-K}, \dots, x_{t-1})
\]

で与えられる。

本モデルは、各ラグ $k \in \{1,\dots,K\}$ に対して状態依存係数行列

\[
\Phi_k(x_{t-k}) \in \mathbb{R}^{p \times p}
\]

をニューラルネットワークにより生成する。予測値は

\[
\hat{x}_t
=
\sum_{k=1}^{K}
\tilde{\Phi}_k(x_{t-k}) x_{t-k}
\]

で定義される。

ここで、構造的 Granger sparsity を表す gate 行列

\[
G_k \in \mathbb{R}^{p \times p}
\]

を導入し、

\[
\tilde{\Phi}_k(x_{t-k})
=
G_k \odot \Phi_k(x_{t-k})
\]

とする。$\odot$ は Hadamard 積である。

実装上、係数テンソルは

\[
[\text{batch}, \text{lag}, \text{target}, \text{source}]
\]

の形を持ち、gate は

\[
[\text{lag}, \text{target}, \text{source}]
\]

の形を持つ。[1](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/models.py)

---

### 2. Granger Non-causality

変数 $x_j$ が変数 $x_i$ を Granger cause しないとは、すべてのラグにおいて $x_j$ から $x_i$ への係数がゼロであることに対応する。

本モデルでは、これは gate により

\[
G_{1,i,j} = G_{2,i,j} = \cdots = G_{K,i,j} = 0
\]

として表現される。

したがって、edge-level の Granger causal graph は

\[
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
\]

で定義する。

実装では `gate_group_norms()` により lag 方向のノルムを計算し、`causal_graph_from_gate()` により二値グラフを得る。[1](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/models.py)

---

### 3. Objective Function

学習目的関数は以下である。

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
+
\lambda_{\mathrm{ngc}}
\mathcal{R}_{\mathrm{ngc}}
\]

ただし、ISTA optimizer を用いる場合、$\mathcal{R}_{\mathrm{ngc}}$ は勾配ステップには含めず、後続の proximal step により処理する。[2](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/training.py)

---

#### 3.1 Prediction Loss

予測損失は平均二乗誤差である。

\[
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{N}
\sum_t
\|x_t - \hat{x}_t\|_2^2
\]

実装では `nn.MSELoss(reduction="mean")` を用いる。[2](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/training.py)

---

#### 3.2 Temporal Smoothness Penalty

GVAR の係数は時点ごとに変化するため、隣接時点間で係数が過度に振動しないよう smoothness penalty を導入する。

absolute mode では、

\[
\mathcal{R}_{\mathrm{smooth}}
=
\frac{1}{|\mathcal{T}|}
\sum_{t \in \mathcal{T}}
\|
\tilde{\Phi}_{t+1} - \tilde{\Phi}_{t}
\|_F^2
\]

を用いる。

relative mode では、

\[
\mathcal{R}_{\mathrm{smooth}}
=
\frac{1}{|\mathcal{T}|}
\sum_{t \in \mathcal{T}}
\frac{
\|
\tilde{\Phi}_{t+1} - \tilde{\Phi}_{t}
\|_F^2
}{
\|\tilde{\Phi}_{t}\|_F^2 + \varepsilon
}
\]

に対応する正規化付き penalty を用いる。

実装では `time_index` を用いて、隣接時点

\[
t_{r+1} - t_r = 1
\]

を満たすペアのみに smoothness penalty を課す。これにより、複数系列・複数 replicate がある場合でも、系列境界をまたいだ smoothing を避ける。[2](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/training.py)

---

### 4. NGC Regularization

#### 4.1 Group Lasso

標準設定では、各 target-source pair $(i,j)$ に対して、lag 方向の gate vector

\[
G_{:,i,j}
=
(G_{1,i,j}, \dots, G_{K,i,j})
\]

を1つのグループとして扱う。

group lasso penalty は

\[
\mathcal{R}_{\mathrm{ngc}}
=
\sum_{i=1}^{p}
\sum_{j=1}^{p}
\left\|
G_{:,i,j}
\right\|_2
\]

である。

この penalty により、ある $(i,j)$ について全ラグの gate が同時にゼロ化される。これは

\[
x_j \not\to x_i
\]

という Granger 非因果性に対応する。[3](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/regularizers.py)

---

#### 4.2 Hierarchical Group Lasso

optional に hierarchical group lasso も利用できる。

実装では lag index 0 を最も古いラグと仮定し、nested prefix group

\[
G_{1:k,i,j}
=
(G_{1,i,j}, \dots, G_{k,i,j})
\]

に対して penalty を課す。

\[
\mathcal{R}_{\mathrm{hier}}
=
\sum_{k=1}^{K}
\sum_{i=1}^{p}
\sum_{j=1}^{p}
\left\|
G_{1:k,i,j}
\right\|_2
\]

この形式では、古いラグほど多くの nested group に含まれるため、古いラグに対してより強い shrinkage がかかる。

注意: この解釈は、入力テンソルの lag index 0 が最古ラグである場合に成立する。[3](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/regularizers.py)

---

### 5. Optimization

本実装では optimizer として `adam` または `ista` を選択できる。[2](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/training.py)

---

#### 5.1 Adam

`adam` を用いる場合、目的関数

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
+
\lambda_{\mathrm{ngc}}
\mathcal{R}_{\mathrm{ngc}}
\]

全体に対して通常の勾配更新を行う。

ただし、この場合 gate は厳密にゼロになりにくく、exact sparsity は保証されない。

---

#### 5.2 ISTA / Proximal Gradient

`ista` を用いる場合、まず smooth な項

\[
\mathcal{L}_{\mathrm{smooth-part}}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}
\mathcal{R}_{\mathrm{smooth}}
\]

に対して勾配ステップを行う。

\[
\theta^{(m+1/2)}
=
\theta^{(m)}
-
\eta
\nabla_\theta
\mathcal{L}_{\mathrm{smooth-part}}
\]

その後、gate parameter $G$ に対して proximal operator を適用する。

\[
G^{(m+1)}
=
\mathrm{prox}_{\eta \lambda_{\mathrm{ngc}} \mathcal{R}_{\mathrm{ngc}}}
\left(
G^{(m+1/2)}
\right)
\]

実装では、通常のネットワークパラメータには SGD step を行い、その後 `ngc.prox_(model.causal_gate, learning_rate)` により gate のみ proximal update を行う。[2](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/training.py)

---

### 6. Proximal Operator for Group Lasso

group lasso の proximal operator は group soft-thresholding である。

各 pair $(i,j)$ について

\[
g_{i,j}
=
(G_{1,i,j}, \dots, G_{K,i,j})
\in \mathbb{R}^{K}
\]

とする。このとき proximal update は

\[
g_{i,j}
\leftarrow
\left(
1
-
\frac{\eta \lambda_{\mathrm{ngc}}}
{
\|g_{i,j}\|_2
}
\right)_+
g_{i,j}
\]

である。

ここで

\[
(a)_+ = \max(a,0)
\]

である。

したがって、

\[
\|g_{i,j}\|_2
\le
\eta \lambda_{\mathrm{ngc}}
\]

ならば、

\[
g_{i,j} = 0
\]

となる。

これにより、Granger graph の edge を厳密にゼロ化できる。[3](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/regularizers.py)

---

### 7. Graph Inference

学習後、係数テンソル

\[
\tilde{\Phi}
\]

を全データに対して計算する。

causal strength は、係数の絶対値を batch および lag 方向に集約して得る。

例えば max aggregation の場合、

\[
S_{i,j}
=
\max_{t,k}
|
\tilde{\Phi}_{k,i,j}(x_{t-k})
|
\]

である。

ただし、最終的な causal graph は coefficient strength ではなく、原則として gate から得る。

\[
\hat{A}_{i,j}
=
\mathbf{1}
\left[
\|G_{:,i,j}\|_2 > \tau
\right]
\]

実装では、`causal_gate` が存在する場合は `causal_graph_from_gate()` を優先し、gate が存在しない場合のみ coefficient strength を thresholding する。[2](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/training.py)[1](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/models.py)

---

### 8. Parameter Groups

optimizer では、coefficient network parameters と gate parameters を分離する。

coefficient network には optional に weight decay を適用する。

\[
\lambda_{\mathrm{wd}}
\|\theta_{\Phi}\|_2^2
\]

一方、gate parameter には weight decay を適用しない。

これは、gate の sparsity を weight decay ではなく proximal operator によって制御するためである。[2](https://msotohoku-my.sharepoint.com/personal/kurita_koya_r6_mso_tohoku_ac_jp/Documents/Microsoft%20Copilot%20Chat%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/training.py)

---

### 9. Summary

本アルゴリズムの特徴は以下である。

1. GVAR により、状態依存的な非線形時系列力学を表現する。
2. static gate により、Granger causal structure と dynamic coefficient を分離する。
3. group lasso により、全ラグにわたる Granger edge を1つの単位として正則化する。
4. proximal gradient により、Granger 非因果性に対応する gate block を厳密にゼロ化する。
5. temporal smoothness penalty により、状態依存係数の過度な時間変動を抑制する。
6. 推論時には coefficient threshold ではなく gate norm に基づいて causal graph を構成する。