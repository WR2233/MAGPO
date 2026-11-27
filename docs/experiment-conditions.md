---
type: Experiment Conditions
title: 実験条件(コード版・ハイパーパラメータ)
description: CoordSum 再現で使った手法ごとのコード版とハイパーパラメータの区分、実行の追跡方法。
tags: [coordsum, reproduction, hyperparameters]
generated: { by: ai:claude, at: 2026-11-27T11:09:07+09:00 }
status: draft
sources:
  - id: params-paper
    resource: https://github.com/WR2233/MAGPO/blob/main/experiment_data/params.csv
    title: 論文のハイパーパラメータ表(著者公開、commit 1db0d0b)
  - id: params-ours
    resource: https://github.com/WR2233/MAGPO/blob/main/analysis/data/ours_params.csv
    title: 自分の実行で使った値(条件ごと)
  - id: overrides
    resource: https://github.com/WR2233/MAGPO/blob/main/jobs/coordsum_overrides.py
    title: params.csv から Hydra override を作るスクリプト
---

# 手法とコード版

| 手法 | コード | ファイル |
|---|---|---|
| MAGPO(採用) | **旧版**(学習側 RNN の隠れ状態を更新しない) | `mava/systems/gpo/anakin/rec_magpo_prefix.py` |
| MAGPO(参考) | 現行(隠れ状態を更新する。commit d2ac38b "fix bugs" 以降) | `rec_magpo.py`(seed 1 のみ) |
| CTDS | 旧版コードベース。[CTDS の実装](/ctds-implementation.md) | `rec_ctds.py` |
| MAPPO / HAPPO / SABLE / MAT | 現行 | `mava/systems/{ppo,sable,mat}/anakin/` |

旧版が採用なのは、現行より論文に近い結果になるため(8x15-100 の最終: 旧版 126、現行 93、論文 120)[^old]。
原因(古い隠れ状態の引き継ぎ vs 文脈の欠落)は仮説のまま未検証。現行コードは seed 1 のみで、この比較は 4 seed 対 1 seed(対照実験は未実施)。

# ハイパーパラメータ

* 学習率・ミニバッチ数・clip・ネットワーク寸法などは `params.csv`(論文の表)の値を、`coordsum_overrides.py` が override として渡す[^p]。
* **GAE λ と価値損失係数 vf coef は、論文の表(0.9 / 0.1)ではなくコード既定値(0.95 / 0.5)**。4 手法で既定値のほうが論文に近かったため(1 seed 時点、8/8 比較)。
* CTDS は MAGPO の行を流用し、`system.alpha=0`(学習側の RL 補助損失なし)。`clip_gpo` は使わない。

# 実行の追跡

* 1 run = 手法×タスク×seed。評価は 122 回(1220 updates、約 2e7 ステップ)、最終スコア = 最後の 10 評価点の平均。
* 設定の検証: `analysis/verify_params_from_logs.py` で、実行ログの設定と `ours_params.csv` を突き合わせる(不一致なし)。
* 実行時の commit は `jobs/wisteria_coordsum.sh` がログと `results/<run>/code_version.txt` に記録する(2026-11-27 以降の実行)。それ以前の実行には記録がない。

[^old]: 条件別の比較は `analysis/data/ours_*.csv` と図 01。
[^p]: 値は `ours_params.csv` に、実際に使った値が条件ごとに残っている。
