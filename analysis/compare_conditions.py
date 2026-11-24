"""実験条件(設定)ごとの CoordSum の結果を、論文の分布と比べる。

指標(すべて生の return。1 run ごとに計算し、論文は 10 seed の分布を作る):
  final : 最後の 10 評価点の平均(最終付近。1点だけより評価ノイズに強い)
  auc   : 全評価点の平均(学習の速さ。立ち上がりが遅いと小さい)
z = (自分の値 - 論文の平均) / 論文の seed 間の標準偏差(1 seed と 10 seed の分布の比較)。
未完了の run(評価点が 122 に満たない)は除外する。

使い方(条件は analysis/common.py の登録簿を既定で全部使う):
  uv run --no-project --with numpy python analysis/compare_conditions.py
  # 条件を指定する場合: --cond "ラベル=csvパス" を繰り返す。--fig で手法×タスクの曲線図も出す。
"""
import argparse
import csv
from collections import defaultdict

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASKS = ["3x10-30", "3x30-50", "5x20-80", "8x15-100"]
N_POINTS = 122
LAST = 10


def load(path):
    tmp = defaultdict(list)
    for r in csv.DictReader(open(path)):
        tmp[(r["system"].replace("magpoold", "magpo"), r["task"], int(r["seed"]))].append((int(r["step_count"]), float(r["mean_episode_return"])))
    out, dropped = {}, []
    for k, v in tmp.items():
        v.sort()
        if len(v) < N_POINTS:
            dropped.append((k, len(v)))
            continue
        out[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v]))
    return out, dropped


def metrics(y):
    return {"final": float(y[-LAST:].mean()), "auc": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=C.PAPER_CSV)
    ap.add_argument("--cond", action="append", help='"ラベル=csvパス"(省略時は登録簿の全条件)')
    ap.add_argument("--fig")
    a = ap.parse_args()

    paper, _ = load(a.paper)
    conds = {}
    cond_args = a.cond or [f"{c['short']}={os.path.join(C.DATA, c['csv'])}" for c in C.CONDITIONS]
    for c in cond_args:
        label, path = c.split("=", 1)
        data, dropped = load(path)
        conds[label] = data
        for k, n in dropped:
            print(f"[除外] {label}: {k[0]} {k[1]} seed{k[2]} は未完了({n}/{N_POINTS}点)")
    systems = ["magpo", "mappo", "happo", "sable", "mat"]
    labels = list(conds)

    # 論文の分布
    pm = defaultdict(lambda: defaultdict(list))
    for (s, t, sd), (_, y) in paper.items():
        for mk, mv in metrics(y).items():
            pm[(s, t, mk)]["v"].append(mv)

    summary = defaultdict(lambda: defaultdict(list))  # (metric, system, label) -> |z| の列
    for mk, title in (("final", f"最終付近(最後の{LAST}評価点の平均)"), ("auc", "学習の速さ(全評価点の平均)")):
        print(f"\n{'=' * 100}\n指標: {title}   z = (自分 - 論文平均) / 論文seed間std\n{'=' * 100}")
        head = f"{'手法':7s} {'タスク':9s} {'論文 mean±std':>16s} | " + " | ".join(f"{l[:18]:>22s}" for l in labels)
        print(head)
        for s in systems:
            for t in TASKS:
                v = np.array(pm[(s, t, mk)]["v"]) if (s, t, mk) in pm else None
                if v is None or len(v) < 2:
                    continue
                cells = []
                for l in labels:
                    k = (s, t, 1)
                    if k in conds[l]:
                        val = metrics(conds[l][k][1])[mk]
                        z = (val - v.mean()) / v.std(ddof=1)
                        summary[(mk, s, l)]["z"].append(z)
                        cells.append(f"{val:8.1f} (z={z:+5.1f})")
                    else:
                        cells.append(f"{'—':>22s}")
                print(f"{s:7s} {t:9s} {v.mean():8.1f}±{v.std(ddof=1):5.1f} | " + " | ".join(f"{c:>22s}" for c in cells))
    print(f"\n{'=' * 100}\n手法ごとの要約: 4タスクの |z| の平均(小さいほど論文に近い) / z の平均(符号=高低の傾向)\n{'=' * 100}")
    for mk in ("final", "auc"):
        print(f"[{mk}]")
        for s in systems:
            row = []
            for l in labels:
                zs = summary.get((mk, s, l), {}).get("z", [])
                row.append(f"{np.mean(np.abs(zs)):5.2f} / {np.mean(zs):+5.2f} (n={len(zs)})" if len(zs) == 4 else "   — (4タスク未完)")
            print(f"  {s:7s} " + " | ".join(f"{l[:12]}: {r}" for l, r in zip(labels, row)))

    if a.fig:
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
        col = {"magpo": "#2a78d6", "mappo": "#1baf7a", "happo": "#eda100", "sable": "#008300", "mat": "#e87ba4"}
        SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
        sy = [s for s in ["mappo", "happo", "sable", "mat", "magpo"] if any((s, t, 1) in conds[l] for l in labels for t in TASKS)]
        fig, axes = plt.subplots(len(TASKS), len(sy), figsize=(3.4 * len(sy), 2.7 * len(TASKS)), sharex=True, facecolor=SURF)
        alphas = np.linspace(0.45, 1.0, len(labels))
        widths = np.linspace(1.2, 2.2, len(labels))
        for i, t in enumerate(TASKS):
            for j, s in enumerate(sy):
                ax = axes[i][j]
                ax.set_facecolor(SURF)
                ax.grid(axis="y", color=GRID, lw=0.8)
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color(GRID)
                ax.tick_params(colors=INK2, labelsize=7, length=0)
                ks = [k for k in paper if k[0] == s and k[1] == t]
                if ks:
                    arr = np.array([paper[k][1] for k in ks]); st = paper[ks[0]][0] / 1e7
                    m, sd = arr.mean(0), arr.std(0, ddof=1)
                    ax.fill_between(st, m - sd, m + sd, color=col[s], alpha=0.15, lw=0)
                    ax.plot(st, m, color=col[s], ls="--", lw=1.1)
                for l, al, lw in zip(labels, alphas, widths):
                    if (s, t, 1) in conds[l]:
                        st2, y = conds[l][(s, t, 1)]
                        ax.plot(st2 / 1e7, y, color=col[s], alpha=al, lw=lw)
                if i == 0:
                    ax.set_title(s.upper(), loc="left", fontsize=10, fontweight="bold", color=INK)
                if j == 0:
                    ax.set_ylabel(f"{t}\nreturn", fontsize=8, color=INK2)
                if i == len(TASKS) - 1:
                    ax.set_xlabel("timesteps (1e7)", fontsize=8, color=INK2)
        from matplotlib.lines import Line2D
        h = [Line2D([0], [0], color=INK2, ls="--", lw=1.1)] + [Line2D([0], [0], color=INK2, alpha=al, lw=lw) for al, lw in zip(alphas, widths)]
        fig.legend(h, ["論文(10 seed 平均、帯=±標準偏差)"] + labels, loc="upper center", ncol=len(h), frameon=False, fontsize=9, labelcolor=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(a.fig, dpi=140, facecolor=SURF)
        print("\nsaved", a.fig)


if __name__ == "__main__":
    main()
