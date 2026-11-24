# CoordSum 再現の解析

論文 Figure (a)(CoordSum)の再現結果を、論文の生データと比べるためのスクリプト群。
リポジトリのルートで `uv run --no-project --with numpy --with matplotlib python analysis/<script>.py` として実行する。

## 流れ
1. `collect_coordsum.py` : 実行結果(`results*/`)や論文データ → 生の return の縦持ち CSV(`analysis/data/`)
2. `make_figures.py`     : 図を一括生成(`figures/01〜04_*.png`)
3. `compare_conditions.py` : 条件ごとの z 値表(final / auc)
4. `make_ours_params.py` / `verify_params_from_logs.py` : 使ったハイパラの記録と、実行ログとの突き合わせ

## 実験条件(`common.py` の登録簿、`data/ours_params.csv` の `condition` 列と同名)
| condition | コード | GAE λ / vf | 手法 | データ |
| --- | --- | --- | --- | --- |
| v1_codedefault | 現行 | 0.95 / 0.5 | 5 手法 | `data/ours_v1_codedefault.csv` |
| v2_paperdefault | 現行 | 0.9 / 0.1 | 5 手法 | `data/ours_v2_paperdefault.csv` |
| old_codedefault | 旧版(学習側 RNN の隠れ状態を更新しない) | 0.95 / 0.5 | MAGPO のみ | `data/ours_old_codedefault.csv` |
| old_paperdefault | 旧版 | 0.9 / 0.1 | MAGPO のみ | `data/ours_old_paperdefault.csv` |

seed 数は条件ごとに異なる(`ours_params.csv` の seed 列と、図の凡例の n を参照)。論文は 10 seed(seed 0〜9)で、`data/paper_coordsum.csv`。
複数 seed の描き方: 曲線は seed 平均 + ±std の帯、棒は seed 平均 + ±std の誤差棒、1 seed は線・棒だけ。

## 結果の更新手順(Wisteria で seed を追加したとき)
```bash
# 1. 結果とログを取得(ブランチごとのディレクトリ。MAGPO=main の 4 手法、MAGPO_old=旧版 MAGPO)
rsync -a wisteria:/work/gg45/g45018/MAGPO/results/      results/
rsync -a wisteria:/work/gg45/g45018/MAGPO_old/results_old/ results_old/
# 2. CSV に集める(新しい seed も自動で入る。MAGPO(旧版)は system 名 magpoold)
R="uv run --no-project --with numpy --with matplotlib python"
$R analysis/collect_coordsum.py ours results     analysis/data/ours_v1_codedefault.csv
$R analysis/collect_coordsum.py ours results_old analysis/data/ours_old_codedefault.csv
# 3. 図と表
$R analysis/make_figures.py
$R analysis/compare_conditions.py
```
評価点が 122 に満たない run(実行中・失敗)は、読み込み時に除外される。

## 図
| ファイル | 内容 |
| --- | --- |
| `01_magpo_conditions.png` | MAGPO の 4 条件 vs 論文(学習曲線 + final/auc) |
| `02_final_vs_paper.png` | 全手法の最終スコア vs 論文(薄=コード既定値、濃=論文既定値) |
| `03_curves_by_task.png` | 手法×タスクの学習曲線(論文の帯に、自分の 2 条件: 薄い線=コード既定値、濃い線=論文既定値。全手法とも現行コード) |
| `04_normalized_overview.png` | Figure (a) 形式(4 タスク平均、タスクごとの最大値で正規化) |

`--set paperdefault|codedefault` で 04 の代表セット(MAGPO=旧版、他=現行コード)を切り替える。
`figures/archive/` は旧版の図(置き換え済み)。

## 指標
- final = 最後の 10 評価点の平均、auc = 全 122 評価点の平均(学習の速さ)。
- z = (自分 − 論文平均) / 論文の seed 間 std。|z| の 4 タスク平均が小さいほど論文に近い(1 seed の偶然で約 0.8)。
- 評価点が 122 に満たない run(未完了)は除外する。
