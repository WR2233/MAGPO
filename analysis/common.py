"""CoordSum 解析の共通部品: 定数・色・実験条件の登録簿・CSV 読み込み・指標。

実験条件(condition)は analysis/data/ours_params.csv の condition 列と同じ名前を使う。
  v1_codedefault   : 現行コード + コード既定ハイパラ (GAE λ=0.95, vf=0.5)   5 手法
  v2_paperdefault  : 現行コード + 論文既定ハイパラ (GAE λ=0.9,  vf=0.1)    5 手法
  old_codedefault  : 旧版 MAGPO(学習側 RNN の隠れ状態を更新しない)+ コード既定   MAGPO のみ
  old_paperdefault : 旧版 MAGPO + 論文既定                                      MAGPO のみ
旧版は system 名 magpoold で保存されるが、読み込み時に magpo に揃える。
"""
import csv
import os
from collections import defaultdict

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PAPER_CSV = os.path.join(DATA, "paper_coordsum.csv")
TASKS = ["3x10-30", "3x30-50", "5x20-80", "8x15-100"]
SYSTEMS = ["magpo", "mappo", "happo", "sable", "mat"]
N_POINTS = 122  # 完了した run の評価点の数
LAST = 10       # final = 最後の LAST 評価点の平均

LABEL = {"magpo": "MAGPO", "mappo": "MAPPO", "happo": "HAPPO", "sable": "SABLE", "mat": "MAT", "ctds": "CTDS"}
COLOR = {"magpo": "#2a78d6", "mappo": "#1baf7a", "happo": "#eda100", "sable": "#008300", "mat": "#e87ba4", "ctds": "#eb6834"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PAPER_COLOR = "#0b0b0b"

CONDITIONS = [
    dict(key="v1_codedefault", csv="ours_v1_codedefault.csv", color="#2a78d6", short="現行+既定",
         label="現行コード・既定値(GAE0.95/vf0.5)", systems=SYSTEMS),
    dict(key="v2_paperdefault", csv="ours_v2_paperdefault.csv", color="#eda100", short="現行+論文既定",
         label="現行コード・論文既定値(GAE0.9/vf0.1)", systems=SYSTEMS),
    dict(key="old_codedefault", csv="ours_old_codedefault.csv", color="#6250d6", short="旧版+既定",
         label="旧版コード・既定値(GAE0.95/vf0.5)", systems=["magpo"]),
    dict(key="old_paperdefault", csv="ours_old_paperdefault.csv", color="#d6336c", short="旧版+論文既定",
         label="旧版コード・論文既定値(GAE0.9/vf0.1)", systems=["magpo"]),
]
COND = {c["key"]: c for c in CONDITIONS}
# 代表セット: MAGPO=旧版、ベースライン=現行コード(5手法とも同じハイパラ区分)
SETS = {
    "paperdefault": {"magpo": "old_paperdefault", "others": "v2_paperdefault"},
    "codedefault": {"magpo": "old_codedefault", "others": "v1_codedefault"},
}


def load_runs(path, complete_only=True):
    """CSV -> {(system, task, seed): (steps[T], returns[T])}(生の return)。完了していない run は除く。"""
    tmp = defaultdict(list)
    for r in csv.DictReader(open(path)):
        s = r["system"].replace("magpoold", "magpo")
        tmp[(s, r["task"], int(r["seed"]))].append((int(r["step_count"]), float(r["mean_episode_return"])))
    out = {}
    for k, v in tmp.items():
        v.sort()
        if complete_only and len(v) < N_POINTS:
            continue
        out[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v]))
    return out


def load_paper():
    return load_runs(PAPER_CSV)


def load_condition(key):
    return load_runs(os.path.join(DATA, COND[key]["csv"]))


def load_set(name):
    """代表セット -> {(system, task, seed): (steps, y)}(MAGPO は旧版、他は現行コード)。"""
    s = SETS[name]
    out = {k: v for k, v in load_condition(s["others"]).items() if k[0] != "magpo"}
    out.update({k: v for k, v in load_condition(s["magpo"]).items() if k[0] == "magpo"})
    return out


def metric(y, kind):
    """final = 最後の LAST 評価点の平均 / auc = 全評価点の平均(学習の速さ)。"""
    return float(y[-LAST:].mean()) if kind == "final" else float(y.mean())


def paper_stats(paper, system, task, kind):
    v = np.array([metric(y, kind) for k, (_, y) in paper.items() if k[0] == system and k[1] == task])
    return v.mean(), v.std(ddof=1)


def style_ax(ax, labelsize=8):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=labelsize, length=0)
