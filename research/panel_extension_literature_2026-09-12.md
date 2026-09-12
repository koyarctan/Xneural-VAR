# XNeuralVARのパネル拡張：先行研究・公開データ調査

調査日：2026年9月12日。対象：ID-POS、都道府県パネル（候補は N=47、T=20）、日本マーケティングサイエンス学会での発表。

今回の成果は文献調査とその報告である。以下の拡張上の論点は、次の議論に持ち越すために記録したもので、実装方針の決定ではない。

## 1. 現行アルゴリズムの理解

確認したのはREADME、slides/slide.tex、models.py、data.py、training.py、regularizers.py、および関連テストである。以下では N を個体数、T を各個体の時点数、p を個体内の変数数、K をラグ次数とする。47都道府県は N=47 であり、直ちに p=47 を意味しない。

### 強み

予測は次の構造を持つ。

\[
\widehat{x}_{i,t}
=\sum_{k=1}^{K}\sum_{j=1}^{p}
G_{k,i,j}\Phi_{k,i,j}(G_{k,i,:}\odot x_{t-k})x_{j,t-k}.
\]

- 非線形な係数生成器により、観測された状態に応じて係数を変化させられる。
- 非負ゲート G が構造の選択を、有効係数 GΦ が符号を含む予測の分解を担う。
- 同じゲートを係数生成器の入力と出力の両方に掛ける。target別のネットワークなので、G=0なら、そのsourceから当該targetへの係数生成経路と直接の乗算経路をともに遮断できる。
- Sparse Group Lassoはエッジ全体と個別ラグを選択する。HGLはラグの階層構造を用いる。ISTAの非負近接更新により、学習パラメータそのものを厳密にゼロ化できる。
- 有効係数を観測別に取り出せるので、共通の構造の下で、どの状態で関係の符号・強弱が変わるかを調べられる。

根拠：[READMEのモデル式と経路遮断](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/README.md:317>)、[forward](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/xneural_var/models.py:248>)、[非負SGL近接更新](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/xneural_var/regularizers.py:218>)。

### パネル拡張を考える上での現状

複数の [T_r,p] 配列をリストで受け取る機能は既にある。個体をまたいだラグ窓は作らず、time_indexに間隔を設けて、別個体間への時間平滑化も避けている。したがって、共通ゲート・共通係数生成器による完全プール学習は、現行APIの範囲に入る。

一方、series_indexを保持していてもモデルには渡していない。同じラグ入力なら、個体が違っても同じ予測・係数になる。個体別の平均水準、構造、同じ状態に対する応答差、個体間の波及を明示的に学ぶ機構はない。状態分布が違うために係数分布が違うことと、個体固有の生成機構が違うことは区別する必要がある。

根拠：[複数系列のデータ構築](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/xneural_var/data.py:24>)、[学習ループ](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/xneural_var/training.py:222>)、[fit関数](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/xneural_var/training.py:367>)。

### 現段階で区別しておくべき解釈

1. **G=0は学習済み予測関数の非依存性の十分条件。** 真のデータ生成過程での非因果性を有限標本から保証する定理ではない。G>0から必ず真のGranger因果性があるという逆向きの主張もできない。MSEで学ぶ対象は主に条件付き平均である。
2. **有効係数は一般には限界効果そのものではない。** 入力が係数生成器も変えるため、入力微分には追加項がある。READMEの微分式からも確認できる。係数の符号と、価格を変化させたときの予測全体の微分の符号が必ず一致するわけではない。
3. **状態依存性と任意の時間変化は異なる。** 現モデルはラグ入力を通じて係数が変化する。同じ状態であっても暦時点だけで関係が変わる機構は、明示的には持たない。
4. **厳密なゼロと調整不要は異なる。** 事後的なグラフ閾値への依存は減らせても、正則化強度、初期値、最適化への依存は残る。slidesのbest F1は、未知グラフに対するλ選択の成功を示す指標ではない。

そのほか、ゲートと係数ネットのスケールの非識別性、pK個のMLPによるパラメータ数、ラグごとの加法構造、予測式に独立した切片がないことも確認した。係数ネットのweight decayは用意されているが、設定の初期値は0である。時間平滑化は同じバッチに入った隣接時点に対して計算するため、短い系列のランダム小バッチでは実効的な平滑化が弱くなりうる。

根拠：[入力微分](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/README.md:355>)、[設定とoptimizer](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/xneural_var/training.py:21>)、[時間平滑化](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/xneural_var/training.py:169>)、[slidesの実験上の課題](<C:/Users/ganga/OneDrive/ドキュメント/Xneural VAR/slides/slide.tex:775>)。

## 2. 優先して読むべき先行研究

### A. アーキテクチャと問題設定が最も近い研究

**Lin, Lei & Michailidis (2024), “A VAE-based Framework for Learning Multi-Level Neural Granger-Causal Connectivity,” TMLR.**

共通グラフと個体別グラフを階層潜在変数として学ぶ。node-centric decoderでは、個体別のエッジ変数を入力ゲートとして用い、共有ニューラルネットで時系列を生成する。連続エッジと二値エッジに対応し、連続版は符号も扱う。ただし付録Eの符号の議論には全体反転の曖昧さがある。[論文本文・式(9)–(10)、付録E](https://arxiv.org/html/2402.16131v1)、[著者の公式実装](https://github.com/GeorgeMichailidis/vae-multi-level-neural-GC-official)。

XNeuralVARとの比較では、入力ゲートによる経路制御と個体間共有が特に近い。一方、このdecoderは状態依存のVAR係数を明示的に出して線形結合する構造ではなく、XNeuralVARの非負SGL/HGL近接更新とも異なる。「複数個体の非線形GC」や「符号を扱えること」だけを新規性にはできない。

標本数にも注意が必要である。本文の20–50時点は、長い軌道から切り出す学習窓の長さを指す。各個体の観測総数が20しかない設定の検証ではない。[論文3.3節・付録A.5/B.4](https://arxiv.org/html/2402.16131v1#S3.SS3)。

**Fisher, Kim, Fredrickson & Pipiras (2022), “Penalized Estimation and Forecasting of Multiple Subject Intensive Longitudinal Data,” Psychometrika, 87, 403–431.**

multi-VARは個体別の線形VAR係数を「共通成分＋個体固有の偏差」に分解し、それぞれを正則化する。近接勾配法を用い、完全プールと個体別推定の中間を扱う。シミュレーションにはT=30も含まれる。[論文本文・式(7)–(9)](https://arxiv.org/html/2007.05052v2)、[著者による実装解説](https://www.icesi.edu.co/CRAN/web/packages/multivar/vignettes/multiVARExample.html)。

これは現コードの最適化方式に接続しやすい設計上の参考であり、線形の主要比較対象でもある。ただし、線形係数を足し合わせる方法を非負ゲートにそのまま移すと、共通エッジを個体ごとに削除できるか、ゼロの意味を保てるかが別問題になる。また「共通成分」は自動的に全員のエッジの共通部分を意味せず、罰則によって定義される。

**Nguyen, Ngo & Sabuncu (2024), “GLACIAL: Granger and Learning-based Causality Analysis for Longitudinal Imaging Studies,” Machine Learning for Biomedical Imaging.**

少数回しか観測されない複数個体を対象とする。共有RNN、入力feature dropout、モデルによる欠測補間を組み合わせ、保持した個体で予測誤差が改善するかを検定する。初稿は2022年、刊行は2024年。[刊行情報](https://arxiv.org/abs/2210.07416)、[本文3節](https://arxiv.org/html/2210.07416v2#S3)。

短い・不規則・欠測のある個体系列の扱いと、個体単位の評価が参考になる。ゲートによる構造選択やVAR型係数の直接解釈を目的とする手法ではない。論文は個体独立、未観測交絡なし、瞬時効果なし、DAGを仮定しているため、フィードバックを含むマーケティングVARに後処理まで無条件に移植することはできない。

**Löwe, Madras, Zemel & Welling (2022), “Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data,” CLeaR, PMLR 177, 509–525.**

サンプルごとにグラフは異なり、相互作用のダイナミクスは共有するという設定。encoderが系列からグラフを推定し、decoderがグラフの下で動的過程を表す。[論文・公式掲載ページ](https://proceedings.mlr.press/v177/lowe22a.html)。

個体ごとに大きなNNを作らず、構造だけを個体別にする発想が参考になる。学習済みencoderで新しい個体のグラフを推定できる点も、顧客数が多い場合に関連する。ただし、XNeuralVARの観測別・ラグ別の有効係数を直接解釈する仕組みとは異なる。

### B. 原型と、短系列をプールする実験の根拠

**Marcinkevičs & Vogt (2021), “Interpretable Models for Granger Causality Using Self-explaining Neural Networks,” ICLR.**

状態依存係数、係数の符号・変動、時間平滑化という原型。原論文の構造選択には係数の集約と安定性に基づく閾値選択がある。パネル拡張の独自性は、この論文とLin et al.の両方に対して示す必要がある。[原論文](https://arxiv.org/abs/2101.07600)。

**Tank, Covert, Foti, Shojaie & Fox, “Neural Granger Causality,” IEEE TPAMI（オンライン2021年、巻号2022年）.**

入力重みの構造的正則化、近接更新、ラグ選択の原型。DREAM3の実験は各データセットがp=100、46反復系列、各21時点で、cMLPの最大ラグを2にしている。[原論文](https://arxiv.org/abs/1802.05842)、[刊行版本文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9739174/)。

これはN=47、T=20に近い規模の短系列学習の例として有用。ただし共通のネットワークを持つ反復系列であり、異質な都道府県パネルと同一条件ではない。「T=20だからNNは不可能」という断定にも、「この実験があるから都道府県別構造を推定できる」という断定にも使えない。

### C. 共通・個体別構造と正則化の追加文献

| 文献 | 調査で確認した要点 | 今回との接点 |
|---|---|---|
| Skripnikov & Michailidis (2019), *Regularized Joint Estimation of Related Vector Autoregressive Models* | group lassoとlassoで関連する線形VARの共通・個体成分を推定。[論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7079674/) | 共通のsupportを促すことと、全個体の係数値を同一にすることの区別。 |
| Crawford et al., *Penalized Subgrouping of Heterogeneous Time Series*（2024年公開プレプリント） | multi-VARに集団内のサブグループ成分を導入。[著者原稿](https://arxiv.org/abs/2409.03085) | 顧客セグメント／地域群単位の動学。刊行情報は今回未確認。 |
| Kim, Fisher & Pipiras, *Joint modeling and inference of multiple-subject high-dimensional sparse vector autoregressive models*（2025年公開プレプリント） | 共通成分の識別条件と、個体ごとの疎性・標本数を反映した推論を扱う。[著者原稿](https://arxiv.org/abs/2510.14044) | 共通＋個体差の分解は、識別条件まで検討対象になる。今回は要旨確認。 |
| Xu & Michailidis, *Joint Learning of Panel VAR models with Low Rank and Sparse Structure*（2025年公開プレプリント） | 共有低ランク構造、個体別重み、疎な固有成分を組み合わせる。[著者原稿](https://arxiv.org/abs/2509.15402) | 個体別パラメータ数を抑える参考。今回は要旨確認で、非線形ゲートへの適用は未検証。 |

これらをすべて実装比較する必要はない。調査上の優先順位は、Lin et al.、multi-VAR、GLACIAL、ACDの順。前二者で構造共有の論点を押さえ、後二者で短系列・新規個体の扱いを補う。

ID-POSを週次化せず購買時刻のイベント列として扱う場合には、Wu et al. (2024), *Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes*も関連する。複数の不規則なイベント系列から、イベント別の相互作用を表す点過程モデルである。ただし、ここでのinstanceは「個々のイベント」であり、「顧客ごとの異質なパネルVAR」とは異なる。現在の等間隔VARを拡張する参考文献とは別の選択肢として記録する。[著者原稿・2節と4節](https://arxiv.org/html/2402.03726v1)。

## 3. パネルとして外せない文献

| 文献 | 参考になる点と適用範囲 |
|---|---|
| Canova & Ciccarelli (2013), *Panel Vector Autoregressive Models: A Survey*, ECB WP 1507 | パネルVARの異質性、個体間依存、パラメータ共有を整理する入口。単なる系列プールと経済学的なパネルVARの違いを確認する。[ECB原稿](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1507.pdf) |
| Nickell (1981), *Biases in Dynamic Models with Fixed Effects* | 短い動学パネルにおける固定効果推定のバイアス。NNに固定効果や個体内中心化を足すだけで解決したと考えないための基礎。古典結果がNNへ同じ形で成立するという主張ではない。[著者所属機関の書誌](https://ora.ox.ac.uk/objects/uuid%3Aa700a01d-f5e9-43c3-b0d6-3c6ad38d4dfa) |
| Dumitrescu & Hurlin (2012), *Testing for Granger non-causality in heterogeneous panels* | 個体ごとのWald統計量を集約する検定。帰無仮説は「全個体で非因果」。棄却は全員共通のエッジを意味しない。短T向けの近似、横断依存向けのblock bootstrapも扱う。[刊行論文](https://www.sciencedirect.com/science/article/pii/S0264999312000491) |
| Juodis, Karavias & Sarafidis (2021), *A homogeneous approach to testing for Granger non-causality in heterogeneous panels* | バイアス補正を取り入れたパネルGC検定。短Tを含む有限標本で、標準的なパネル検定をそのまま正解扱いしないための参考。[刊行論文](https://link.springer.com/article/10.1007/s00181-020-01970-9) |
| Minorics et al. (2022), *Testing Granger Non-Causality in Panels with Cross-Sectional Dependencies*, AISTATS | 個体別p値を集約して横断依存を考慮する。共通ショックがありうる都道府県や同一チェーン店舗での検定比較に関係する。ゲートにp値を与える方法ではない。[公式論文](https://proceedings.mlr.press/v151/minorics22a.html) |

論文比較でいう「共通構造」は少なくとも、全員に存在するエッジ、平均・代表的な効果、個体間のエッジ存在確率、共通の生成関数を区別する必要がある。各文献が推定しているものを合わせずにグラフ精度を比較すると、異なる対象を評価してしまう。

N=47、T=20、K=2なら学習窓は47×18=846個作れる。ただし846個の独立標本という意味ではなく、個体ごとには18個の遷移しかない。共有構造が強い部分と個体固有部分では、情報量が大きく違う。個体別の自由なNNと自由なグラフを一度に学ぶ根拠は、今回確認した文献からは得られていない。

## 4. マーケティング研究との接続と公開データ

### 研究上の接続

**Pauwels, Hanssens & Siddarth (2002), “The Long-Term Effects of Price Promotions on Category Incidence, Brand Choice, and Purchase Quantity,” Journal of Marketing Research, 39(4), 421–439.**

スキャナーパネル由来の週次データから、価格販促と購買の各構成要素の動的関係を調べる研究。販促直後の反応、後続期間の調整、累積的な反応を区別する題材として参考になる。[著者公開の刊行論文](https://marketingandmetrics.com/wp-content/uploads/2020/06/19.-The-long-term-effects.pdf)。

今回への接続としては、「あるブランドの購買・価格・販促が、翌週以降のどの変数の予測に関係するか」「その関係が店舗／顧客の状態によりどう変わるか」が考えられる。これらは研究課題の候補であり、データにその効果が存在するという主張ではない。

週次VARの過去価格から翌週売上への関係は、同じ週の価格弾力性と異なる。また、観測価格やクーポン利用は需要に反応して決まる可能性がある。係数の符号をそのまま販促介入の因果効果と呼ぶには追加の識別が必要になる。

### データ候補

以下の規模は公表仕様であり、欠測や採用品目を絞った後の実効N・Tではない。今回は実データのダウンロード・整形は行っていない。

| 候補 | パネルの組み方と規模 | 変数・本手法との接点 | 利用条件・残る確認 |
|---|---|---|---|
| **Dominick’s Finer Foods（第一候補）** | 店舗×週。研究用データでは約93店舗、全体の週インデックスは最大400週。採用商品と共通観測期間で変わる。 | UPC別販売数量、価格、販促コード、利益率、店舗属性。店舗を個体、共通するブランド等を変数として構成すれば、変数間の関係と店舗間異質性を扱える。 | 無料公開、学術研究目的限定、Kilts Centerの謝辞が必要。[公式配布元](https://www.chicagobooth.edu/research/kilts/research-data/dominicks)、[マニュアル](https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter.aspx)、[93店舗・400週を記載した公式サイト掲載論文](https://www.chicagobooth.edu/-/media/Research/Kilts/docs/Snir-and-Levy-JACR-2021)。 |
| **dunnhumby The Complete Journey（ID-POSを優先する場合）** | 世帯×週等に集計。元の公開版は約2,500世帯、2年間。週次化すれば暦上は約104週。 | 購買履歴、商品カテゴリ、世帯属性、一部世帯のマーケティング接触履歴。顧客ごとの購買関係・異質性に近い。 | 公式に学術研究用途を案内する公開データ。提供元は実データに基づく「representation」と説明。今回、公式検索本文で仕様を確認したが、直接ページ取得はタイムアウトし、配布ファイルと同梱条件は未確認。[公式案内](https://www.dunnhumby.com/source-files/)。 |
| dunnhumby Breakfast at the Frat（小さく始める予備候補） | 店舗×週×商品。156週。店舗数は今回、一次資料では確定していない。 | 販売数量、購買世帯数、来店、支出、通常価格・棚価格、販促サポート。週次の価格・販促分析向けに設計されている。 | 公開教材データ。Complete Journeyと同じく、採用前にファイルと同梱条件を確認する。[公式案内](https://www.dunnhumby.com/source-files/)。 |
| **観光庁・宿泊旅行統計（国内地域パネルを優先する場合）** | 都道府県×月。N=47として構成可能。複数年の月次を使えば、年次T=20より長い系列を得られる。 | 日本人／外国人宿泊、稼働率等。観光需要の動的関係に結び付けられる。採用するp変数は別途選定する。 | 公式サイトとe-StatでExcel等を公開。期間・定義変更・季節性・コロナ期の扱いを確認する。今回は共通期間と欠測の突合せは未実施。[観光庁](https://www.mlit.go.jp/kankocho/tokei_hakusyo/shukuhakutokei.html)、[e-Statの月次データ説明](https://www.e-stat.go.jp/stat-search/files?cycle=1&data=1&layout=dataset&metadata=1&page=1&stat_infid=000040247441&tclass1val=0&toukei=00601020&tstat=000001079597)。 |

Complete Journeyには別にRパッケージ版があり、こちらの説明は2,469世帯・1年間である。元の2年版と混同しない。[パッケージ提供者の説明](https://bradleyboehmke.github.io/completejourney/)。

適合性についての現時点の判断は、手法の初回実証にはDominick’s、顧客単位を研究の中心に据えるならComplete Journey、国内の地域的意義を重視するなら宿泊旅行統計である。いずれも実グラフの正解は既知ではないため、構造回復の評価は合成データで行い、実データでは予測・再標本化での安定性・解釈を評価する役割分担が自然である。

ID-POSでは「購買なしのゼロ」と「観測されていないこと」を区別する。買っていない週には購入単価が定義できないため、価格を一律ゼロ補完する設計は避ける必要がある。Dominick’sでも、商品入替えや未取扱いと真の販売ゼロの区別、チェーン共通販促・季節性による横断依存が論点になる。

## 5. 次の議論へ持ち越す項目

調査範囲内では、**パネルの非線形GC・構造共有・入力ゲート**には明確な先行研究がある。一方、XNeuralVARの**target・lag別の状態依存VAR係数、入力と係数の二重ゲート、非負SGL/HGLの厳密な近接更新**をそろえた異質パネル版は確認できなかった。これは今回の検索範囲での確認結果であって、先行研究が存在しないことの証明ではない。

今後の議論では次を決める必要がある。

1. 共通にする対象はグラフか、係数生成関数か、その両方か。
2. 個体差は平均水準、係数の応答、エッジの存在、セグメント差のどこまで扱うか。
3. 個体差や共通因子を追加しても、ゲートゼロ時の経路遮断が保たれるか。
4. 短T、固定効果、横断依存、欠測、同時点の販促反応をどこまでモデル化するか。
5. λとモデル容量を未知グラフでどう選び、構造と符号の安定性をどう評価するか。

この調査ではモデルや学習コードは変更していない。

## 調査範囲の記録

検索語は self-explaining / neural Granger / multi-entity / multi-subject / multi-level / panel / multi-VAR / shared dynamics / structured sparsity、および小売スキャナー・世帯購買・都道府県月次データに関連する語を使用した。主要な説明は論文本文、著者公開原稿、公式実装、データ提供元を根拠とした。2025年の追加文献は要旨確認として明示した。OpenReviewの直接取得が認証画面となった論文はarXiv本文と著者公式実装で照合した。商用サービス向けの規約を公開教材データの規約とみなしてはいない。
