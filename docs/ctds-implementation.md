---
type: Implementation Note
title: CTDS の実装(論文の記述・経緯・差)
description: 論文の 1 文の記述から CTDS をどう実装したか、前版(KL 常時)と最終版(KL なし)の違い、論文との差。
tags: [ctds, magpo, implementation]
generated: { by: ai:claude, at: 2026-11-27T11:09:07+09:00 }
status: draft
sources:
  - id: paper
    resource: https://arxiv.org/abs/2507.18059
    title: MAGPO 論文(CTDS は "vanilla implementation of on-policy CTDS" として 1 文のみ記述)
  - id: code
    resource: https://github.com/WR2233/MAGPO/blob/main/mava/systems/gpo/anakin/rec_ctds.py
    title: CTDS の実装(rec_magpo_prefix.py のコピーから guider 損失を変更)
  - id: data
    resource: https://github.com/WR2233/MAGPO/blob/main/analysis/data/ours_ctds2_codedefault.csv
    title: 最終版 CTDS の結果(4 seed)
---

# 論文の記述

「on-policy CTDS の素朴な実装は、MAGPO から二重 clip・マスク・RL 補助損失を外したものとみなせる」(1 文のみ。ハイパーパラメータ・損失の詳細は記載なし)[^paper]。
CTDS の定義: guider は拘束されず自由に学習し、student(分散側)が guider に知識蒸留する。

# 実装(最終版)

`rec_magpo_prefix.py`(MAGPO 旧版)とのコード上の差分は guider の損失の 2 点だけ[^code]:

* 二重 clip を外す: `clipped_ratio = ratio`(普通の PPO の比率)。
* guider の KL 項を外す: `kl_loss = 0`。

設定(コードは変えない):

* 学習側の RL 補助損失は `system.alpha=0` で外す。学習側の損失は `alpha*PPO + KL(guider‖learner)` のコードのまま、alpha=0 により KL だけが残る。

ハイパーパラメータは MAGPO の行を流用(GAE λ/vf はコード既定値)。

# 経緯

1. **前版**: 「マスクを外す」を「KL を常に guider にかける」と解釈。結果は MAGPO とほぼ同じ(5x20-80 で 121)。guider が学習側に引き寄せられるため、論文の CTDS の定義(guider は自由)に反していた。
2. **最終版**: guider の KL を外した。論文の傾向に近づいた(5x20-80 で 4.5)。

# 論文との差

最終スコア(論文→自分): 3x10-30 は 89→116、3x30-50 は 69→53、5x20-80 は 18→4、8x15-100 は 22→6。
差の候補(未検証): 論文の CTDS のハイパーパラメータ、蒸留損失の形(KL の向き)、guider の学習の詳細。
自分の CTDS は seed 間のばらつきが小さい(std 0.1〜6)のに対し、論文は大きい(11〜16)。

[^paper]: 論文の記述は引用した 1 文のみで、詳細は確認できていない。
[^code]: 実装の確認: 実行ログで 16 run すべてが `alpha: 0` であることを確認済み(`verify_params_from_logs.py`)。
