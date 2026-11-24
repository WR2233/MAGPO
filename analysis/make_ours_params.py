"""自分の実行で使ったハイパラを、experiment_data/params.csv と同じ列形式で保存する。

params.csv の CoordSum 行(= 実際に Hydra override として渡した値)に、実験条件の列を追加する。
  condition     : 実験条件名
  code          : current(現行コード) / old(学習側RNNの隠れ状態を更新しない旧版)
  gae_lambda    : 実際に使った GAE λ(params.csv に無い列。条件で変える)
  vf_coef       : 実際に使った value 係数
  system_name   : 実行に使った手法名(旧版は magpoold。params.csv 上の元の手法名は params_system)
使い方: python3 analysis/make_ours_params.py
"""
import csv
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "experiment_data", "params.csv")
OUT = os.path.join(ROOT, "analysis", "data", "ours_params.csv")
TASKS = ["3x10", "3x30", "5x20", "8x15"]
# (condition, code, gae, vf, 手法リスト, 結果ディレクトリ)
CONDS = [
    ("v1_codedefault", "current", 0.95, 0.5, ["magpo", "mappo", "happo", "sable", "mat"]),
    ("v2_paperdefault", "current", 0.9, 0.1, ["magpo", "mappo", "happo", "sable", "mat"]),
    ("old_codedefault", "old", 0.95, 0.5, ["magpo"]),
    ("old_paperdefault", "old", 0.9, 0.1, ["magpo"]),
]

rows = [r for r in csv.DictReader(open(SRC)) if r["env_name"] == "coordsum"]
with open(SRC) as f:
    cols = next(csv.reader(f))
extra = ["condition", "code", "gae_lambda", "vf_coef", "params_system", "seed"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols + extra)
    w.writeheader()
    for cond, code, gae, vf, systems in CONDS:
        for s in systems:
            for t in TASKS:
                r = next(x for x in rows if x["system_name"] == s and x["task"] == t)
                o = dict(r)
                o.update(condition=cond, code=code, gae_lambda=gae, vf_coef=vf, params_system=s, seed=1)
                if code == "old":
                    o["system_name"] = "magpoold"
                w.writerow(o)
print("wrote", OUT)
