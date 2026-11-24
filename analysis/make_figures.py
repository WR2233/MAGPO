"""CoordSum 再現の図を一括で作る(リポジトリのルートで実行)。

  01_magpo_conditions.png : MAGPO の 4 条件(現行/旧版 × 既定/論文既定)vs 論文。学習曲線 + final/auc の棒
  02_final_vs_paper.png   : 全手法の最終スコア(final)を、タスクごとに論文と並べた棒グラフ
  03_curves_by_task.png   : 手法×タスクの学習曲線。論文(10 seed 平均±std の帯)に自分の線を重ねる
  04_normalized_overview.png : 論文 Figure (a) 形式(4 タスク平均・タスクごとの最大値で正規化)。論文 | 自分

使い方:
  uv run --no-project --with numpy --with matplotlib python analysis/make_figures.py
  uv run --no-project --with numpy --with matplotlib python analysis/make_figures.py --set codedefault --only 3,4

--set は 03/04 で使う「代表セット」(common.SETS):
  paperdefault : MAGPO=旧版+論文既定、他手法=現行コード+論文既定(既定)
  codedefault  : MAGPO=旧版+既定、他手法=現行コード+既定
データは analysis/data/ の CSV(collect_coordsum.py で作る。条件の登録簿は common.py)。
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]


def fig_conditions(paper, out):
    """MAGPO の 4 条件 vs 論文。"""
    conds = [(c, C.load_condition(c["key"])) for c in C.CONDITIONS]
    fig = plt.figure(figsize=(16, 10), facecolor=C.SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.38, wspace=0.25)

    for j, t in enumerate(C.TASKS):
        ax = fig.add_subplot(gs[0, j]); C.style_ax(ax)
        runs = [y for k, (_, y) in paper.items() if k[0] == "magpo" and k[1] == t]
        st = next(s for k, (s, _) in paper.items() if k[0] == "magpo" and k[1] == t)
        m, sd = np.mean(runs, 0), np.std(runs, 0, ddof=1)
        ax.fill_between(st / 1e7, m - sd, m + sd, color=C.PAPER_COLOR, alpha=0.13, lw=0)
        ax.plot(st / 1e7, m, color=C.PAPER_COLOR, ls="--", lw=1.4)
        for c, d in conds:
            k = ("magpo", t, 1)
            if k in d:
                ax.plot(d[k][0] / 1e7, d[k][1], color=c["color"], lw=1.7)
        ax.set_title(f"CoordSum-{t}", loc="left", color=C.INK, fontsize=10, fontweight="bold")
        ax.set_xlabel("timesteps (1e7)", color=C.INK2, fontsize=8)
        if j == 0:
            ax.set_ylabel("Mean episode return", color=C.INK2, fontsize=9)

    nb, w = len(conds) + 1, 0.16
    for col, (kind, title) in enumerate([("final", f"最終スコア(最後の{C.LAST}評価点の平均)"), ("auc", "学習の速さ(全評価点の平均)")]):
        ax = fig.add_subplot(gs[1, col * 2:(col + 1) * 2]); C.style_ax(ax)
        x = np.arange(len(C.TASKS))
        stats = [C.paper_stats(paper, "magpo", t, kind) for t in C.TASKS]
        pm, ps = [a for a, _ in stats], [b for _, b in stats]
        ax.bar(x - (nb - 1) / 2 * w, pm, w * 0.9, yerr=ps, color="#a8a8a8", ecolor=C.INK2, error_kw=dict(lw=1, capsize=2))
        means = [f"論文 {np.mean(pm):.1f}"]
        for i, (c, d) in enumerate(conds):
            v = [C.metric(d[("magpo", t, 1)][1], kind) if ("magpo", t, 1) in d else np.nan for t in C.TASKS]
            xs = x + (i + 1 - (nb - 1) / 2) * w
            ax.bar(xs, v, w * 0.9, color=c["color"])
            for xx, vv in zip(xs, v):
                ax.text(xx, vv + 1, f"{vv:.0f}", ha="center", va="bottom", fontsize=7, color=C.INK2)
            means.append(f"{c['short']} {np.nanmean(v):.1f}")
        ax.set_title(f"{title}\n4タスク平均: " + "  |  ".join(means), loc="left", color=C.INK, fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(C.TASKS, color=C.INK2, fontsize=9)
        ax.set_ylabel("return", color=C.INK2, fontsize=9)
        ax.set_ylim(0, max(pm) * 1.22)

    h = [Line2D([0], [0], color=C.PAPER_COLOR, ls="--", lw=1.4), Patch(color="#a8a8a8")] + [Line2D([0], [0], color=c["color"], lw=2) for c, _ in conds]
    lb = ["論文(10 seed 平均、帯・誤差棒=±std)", "論文(棒)"] + [c["label"] + "・1 seed" for c, _ in conds]
    fig.legend(h, lb, loc="upper left", bbox_to_anchor=(0.05, 0.955), ncol=2, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.suptitle("MAGPO @ CoordSum: 4条件 vs 論文", x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold", color=C.INK)
    fig.subplots_adjust(top=0.82, bottom=0.07, left=0.06, right=0.98)
    fig.savefig(out, dpi=150, facecolor=C.SURF)
    print("saved", out)


def fig_final_all(paper, out):
    """全手法の最終スコア。薄い色=コード既定値、濃い色=論文既定値(MAGPO は旧版コード)。"""
    pairs = {  # 手法 -> (既定値の条件, 論文既定値の条件)
        s: (("old_codedefault", "old_paperdefault") if s == "magpo" else ("v1_codedefault", "v2_paperdefault")) for s in C.SYSTEMS
    }
    data = {k: C.load_condition(k) for k in {c for p in pairs.values() for c in p}}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), facecolor=C.SURF)
    w = 0.26
    for ax, t in zip(axes, C.TASKS):
        C.style_ax(ax)
        for i, s in enumerate(C.SYSTEMS):
            m, sd = C.paper_stats(paper, s, t, "final")
            ax.bar(i - w, m, w * 0.9, yerr=sd, color="#a8a8a8", ecolor=C.INK2, error_kw=dict(lw=1, capsize=2))
            for j, (ck, al) in enumerate(zip(pairs[s], (0.45, 1.0))):
                k = (s, t, 1)
                if k in data[ck]:
                    v = C.metric(data[ck][k][1], "final")
                    ax.bar(i + j * w, v, w * 0.9, color=C.COLOR[s], alpha=al)
                    ax.text(i + j * w, v + 2, f"{v:.0f}", ha="center", fontsize=7, color=C.INK2)
        ax.set_xticks(range(len(C.SYSTEMS))); ax.set_xticklabels([C.LABEL[s] for s in C.SYSTEMS], fontsize=8, color=C.INK2)
        ax.set_title(f"CoordSum-{t}", loc="left", fontsize=10, fontweight="bold", color=C.INK)
        ax.set_ylim(0, 180)
    axes[0].set_ylabel(f"最終スコア(最後の{C.LAST}評価点の平均)", fontsize=9, color=C.INK2)
    h = [Patch(color="#a8a8a8"), Patch(color=C.INK2, alpha=0.45), Patch(color=C.INK2)]
    fig.legend(h, ["論文(10 seed 平均±std)", "自分: コード既定値(MAGPO は旧版コード)", "自分: 論文既定値(MAGPO は旧版コード)"],
               loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=140, facecolor=C.SURF)
    print("saved", out)


def fig_curves(paper, ours, set_name, out):
    """手法×タスクの学習曲線。"""
    fig, axes = plt.subplots(len(C.TASKS), len(C.SYSTEMS), figsize=(3.4 * len(C.SYSTEMS), 2.6 * len(C.TASKS)), sharex=True, facecolor=C.SURF)
    for i, t in enumerate(C.TASKS):
        ymax = max(max(y.max() for k, (_, y) in paper.items() if k[0] == s and k[1] == t) for s in C.SYSTEMS) * 1.05
        for j, s in enumerate(C.SYSTEMS):
            ax = axes[i][j]; C.style_ax(ax)
            runs = [(st, y) for k, (st, y) in paper.items() if k[0] == s and k[1] == t]
            st, arr = runs[0][0], np.array([y for _, y in runs])
            m, sd = arr.mean(0), arr.std(0, ddof=1)
            ax.fill_between(st / 1e7, m - sd, m + sd, color=C.COLOR[s], alpha=0.18, lw=0)
            ax.plot(st / 1e7, m, color=C.COLOR[s], ls="--", lw=1.2)
            if (s, t, 1) in ours:
                ax.plot(ours[(s, t, 1)][0] / 1e7, ours[(s, t, 1)][1], color=C.COLOR[s], lw=1.8)
            ax.set_ylim(0, ymax)
            if i == 0:
                ax.set_title(C.LABEL[s], loc="left", fontsize=10, fontweight="bold", color=C.INK)
            if j == 0:
                ax.set_ylabel(f"{t}\nreturn", fontsize=8, color=C.INK2)
            if i == len(C.TASKS) - 1:
                ax.set_xlabel("timesteps (1e7)", fontsize=8, color=C.INK2)
    s = C.SETS[set_name]
    note = f"MAGPO={C.COND[s['magpo']]['label']} / 他手法={C.COND[s['others']]['label']}"
    h = [Line2D([0], [0], color=C.INK2, ls="--", lw=1.2), Line2D([0], [0], color=C.INK2, lw=1.8)]
    fig.legend(h, ["論文: 10 seed 平均(帯 = ±std)", f"自分: 1 seed({note})"], loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140, facecolor=C.SURF)
    print("saved", out)


def fig_normalized(ours, set_name, out):
    """論文 Figure (a) 形式。正規化は plot_coordsum.py と同じ関数を使う(タスクごとの最大値で割る)。"""
    import plot_coordsum as pc  # 同じ analysis/ にある

    paper = pc.load(C.PAPER_CSV)
    ours_pc = {k: v for k, v in ours.items()}
    params = pc.norm_params(paper, "maxmean")
    s = C.SETS[set_name]
    titles = {"paper": "論文(10 seed)", "ours": f"自分(1 seed): MAGPO={C.COND[s['magpo']]['short']} / 他={C.COND[s['others']]['short']}"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, facecolor=C.SURF)
    for ax, (name, data) in zip(axes, (("paper", paper), ("ours", ours_pc))):
        pc.style(ax, titles[name])
        pc.draw(ax, pc.curves(data, params))
    axes[0].set_ylabel("Mean episode return (normalized: 各タスクの最大値で割る)", color=C.INK2, fontsize=9)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=len(h), frameon=False, fontsize=9, labelcolor=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=C.SURF)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="paperdefault", choices=list(C.SETS))
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", default="1,2,3,4", help="作る図の番号(カンマ区切り)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    want = set(a.only.split(","))
    paper = C.load_paper()
    ours = C.load_set(a.set) if want & {"3", "4"} else None
    if "1" in want:
        fig_conditions(paper, os.path.join(a.outdir, "01_magpo_conditions.png"))
    if "2" in want:
        fig_final_all(paper, os.path.join(a.outdir, "02_final_vs_paper.png"))
    if "3" in want:
        fig_curves(paper, ours, a.set, os.path.join(a.outdir, "03_curves_by_task.png"))
    if "4" in want:
        fig_normalized(ours, a.set, os.path.join(a.outdir, "04_normalized_overview.png"))


if __name__ == "__main__":
    main()
