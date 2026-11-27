---
type: Results Summary
title: CoordSum 再現結果のまとめ(4 seed)
description: 5 手法+CTDS の最終スコアと論文との差(z)、再現できた/できなかった点。
tags: [coordsum, results, reproduction]
generated: { by: ai:claude, at: 2026-11-27T11:09:07+09:00 }
status: draft
sources:
  - id: ours-data
    resource: https://github.com/WR2233/MAGPO/tree/main/analysis/data
    title: 自分の結果(CSV。MAGPO は ours_old_codedefault.csv、CTDS は ours_ctds2_codedefault.csv)
  - id: paper-data
    resource: https://github.com/WR2233/MAGPO/blob/main/analysis/data/paper_coordsum.csv
    title: 論文の生データ(10 seed)
  - id: figures
    resource: https://github.com/WR2233/MAGPO/blob/main/analysis/make_figures.py
    title: 図 01〜06 の生成コード(画像は figures/、git 管理外)
---

自分 4 seed(MAGPO・CTDS は旧版コードベース、他は現行コード。GAE λ/vf はコード既定値)対 論文 10 seed。
z は Welch 型(平均の差 / sqrt(論文var/10 + 自分var/4))。

# 最終スコア(最後の 10 評価点の平均。論文 → 自分、括弧内は z)

| 手法 | 3x10-30 | 3x30-50 | 5x20-80 | 8x15-100 |
|---|---|---|---|---|
| MAGPO(旧版) | 152→150 (-1.8) | 154→152 (-1.1) | 149→140 (-1.5) | 120→126 (+1.1) |
| MAPPO | 154→158 (+1.7) | 156→152 (-0.7) | 136→132 (-0.6) | 115→124 (+1.0) |
| HAPPO | 152→153 (+0.2) | 125→**90 (-14.3)** | 117→115 (-0.3) | 69→65 (-0.9) |
| SABLE | 134→131 (-0.5) | 111→110 (-0.2) | 46→38 (-0.5) | 26→33 (+0.8) |
| MAT | 50→54 (+0.5) | 76→87 (+1.9) | 74→69 (-1.0) | 51→44 (-1.9) |
| CTDS(guider KL なし) | 89→116 (+5.0) | 69→53 (-3.0) | 18→4 (-3.7) | 22→6 (-3.3) |

# 判定

* **おおむね再現**: MAGPO(旧版)、MAPPO、SABLE、MAT、HAPPO の 3 タスク。20 条件(CTDS を除く 5 手法×4 タスク)中 15 条件が論文の 1 標準偏差以内。MAGPO と MAPPO の大小関係も論文と同じ(5x20-80 のみ MAGPO が明確に上)。
* **再現できていない**: HAPPO の 3x30-50(論文 125 に対し 90。原因未解明。MAPPO は同じ設定で再現できているので HAPPO 固有の可能性)。
* **定性的に一致、数値は不一致**: CTDS。MAGPO より大きく劣り、5x20-80 と 8x15-100 でほぼ学習しない点は論文と同じだが、z は +5.0〜-3.7。[CTDS の実装](/ctds-implementation.md)参照。
* **未解明の差**: MAT の 3x10-30 は最終値は一致するが序盤の立ち上がりが論文より遅い。

# 注意

* MAGPO は旧版コードの結果。現行コードは seed 1 のみで論文に届かない(対照実験は未実施)。
* 自分の seed 数は 4(論文は 10)。HAPPO も 4 seed で揃った。
