"""CoordSum 再現の図を一括で作る(リポジトリのルートで実行)。複数 seed のデータにも対応する。

  01_magpo_conditions.png : MAGPO の 4 条件(現行/旧版 × 既定/論文既定)vs 論文。学習曲線 + final/auc の棒
  02_final_vs_paper.png   : 全手法(CTDS 含む)の最終スコア(final)を、タスクごとに論文と並べた棒グラフ
  03_curves_by_task.png   : 手法×タスクの学習曲線。論文(10 seed 平均±std の帯)に、自分の 2 条件(コード既定値=薄い線、論文既定値=濃い線)を重ねる。
                            MAGPO・他手法は現行コード。CTDS は旧版コードベースでコード既定値のみ(論文既定値の実行なし)
  04_normalized_overview.png : 論文 Figure (a) 形式(4 タスク平均・タスクごとの最大値で正規化)。論文 | 自分(CTDS 含む)
  05_repro_curves.png     : 採用条件(MAGPO=旧版コード、他=現行コード、全手法コード既定値、CTDS=旧版コードベース)の学習曲線を論文と比較
  06_repro_final.png      : 同条件の最終スコア(棒=seed 平均、点=各 seed、上段=平均・n・論文との z)

自分の結果が複数 seed のとき: 曲線は seed 平均 + ±std の帯、棒は seed 平均 + ±std の誤差棒、
1 seed のときは線・棒だけ(誤差なし)。凡例に seed 数(n)を出す。

使い方:
  uv run --no-project --with numpy --with matplotlib python analysis/make_figures.py
  uv run --no-project --with numpy --with matplotlib python analysis/make_figures.py --set paperdefault --only 3,4

--set は 04 で使う「代表セット」(common.SETS):
  codedefault  : MAGPO=旧版+既定、他手法=現行コード+既定(既定。複数 seed の本命)
  paperdefault : MAGPO=旧版+論文既定、他手法=現行コード+論文既定
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

# 図で扱う手法(論文の凡例の順)。CTDS は旧版コードベース(guider に KL なし。alpha=0)で、コード既定値の実行のみ
FIG_SYSTEMS = ["magpo", "ctds", "mappo", "happo", "sable", "mat"]
CTDS_KEY = "ctds_codedefault"
CTDS_CSV = "ours_ctds2_codedefault.csv"
# 手法 -> (コード既定値の条件, 論文既定値の条件)。MAGPO は旧版コード、他手法は現行コード。CTDS は論文既定値の実行なし(None)
PAIRS = {s: (("old_codedefault", "old_paperdefault") if s == "magpo" else ("v1_codedefault", "v2_paperdefault")) for s in C.SYSTEMS}
PAIRS["ctds"] = (CTDS_KEY, None)
# 全手法とも現行コード(MAGPO も隠れ状態を更新する最新版)。CTDS のみ旧版コードベース
CURRENT_PAIRS = {s: ("v1_codedefault", "v2_paperdefault") for s in C.SYSTEMS}
CURRENT_PAIRS["ctds"] = (CTDS_KEY, None)
# 採用条件: MAGPO=旧版コード、他=現行コード(全てコード既定値)、CTDS=旧版コードベース
REPRO = {s: ("old_codedefault" if s == "magpo" else "v1_codedefault") for s in C.SYSTEMS}
REPRO["ctds"] = CTDS_KEY


def load_cond(key):
    """条件のデータ。CTDS(登録簿に無い)は専用の CSV から読む。"""
    return C.load_runs(os.path.join(C.DATA, CTDS_CSV)) if key == CTDS_KEY else C.load_condition(key)


def n_of(data, systems):
    """描く手法の seed 数(手法ごとの最小値)。"""
    ns = [len({k[2] for k in data if k[0] == s and k[1] == t}) for s in systems for t in C.TASKS]
    ns = [n for n in ns if n > 0]
    return min(ns) if ns else 0


def mean_bar(ax, x, v, w, color, alpha=1.0, label=True):
    """seed 平均の棒。複数 seed なら ±std の誤差棒、値を上に表示。v: 各 seed の値の配列(None なら何もしない)。"""
    if v is None:
        return
    m = float(np.mean(v))
    err = float(np.std(v, ddof=1)) if len(v) > 1 else None
    ax.bar(x, m, w * 0.9, yerr=err, color=color, alpha=alpha, ecolor=C.INK2, error_kw=dict(lw=1, capsize=2))
    if label:
        ax.text(x, m + (err or 0) + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=7, color=C.INK2)


def paper_runs(paper, s, t):
    ks = [k for k in paper if k[0] == s and k[1] == t]
    return paper[ks[0]][0], np.array([paper[k][1] for k in ks])


def fig_conditions(paper, out):
    """MAGPO の 4 条件 vs 論文。"""
    conds = [(c, C.load_condition(c["key"])) for c in C.CONDITIONS]
    fig = plt.figure(figsize=(16, 10), facecolor=C.SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.38, wspace=0.25)

    for j, t in enumerate(C.TASKS):
        ax = fig.add_subplot(gs[0, j]); C.style_ax(ax)
        st, arr = paper_runs(paper, "magpo", t)
        m, sd = arr.mean(0), arr.std(0, ddof=1)
        ax.fill_between(st / 1e7, m - sd, m + sd, color=C.PAPER_COLOR, alpha=0.13, lw=0)
        ax.plot(st / 1e7, m, color=C.PAPER_COLOR, ls="--", lw=1.4)
        for c, d in conds:
            sk = C.stack(d, "magpo", t)
            if sk:
                C.band(ax, sk[0], sk[1], c["color"])
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
            xs = x + (i + 1 - (nb - 1) / 2) * w
            vals = [C.seed_metrics(d, "magpo", t, kind) for t in C.TASKS]
            for xx, v in zip(xs, vals):
                mean_bar(ax, xx, v, w, c["color"])
            means.append(f"{c['short']} {np.nanmean([np.mean(v) if v is not None else np.nan for v in vals]):.1f}")
        ax.set_title(f"{title}\n4タスク平均: " + "  |  ".join(means), loc="left", color=C.INK, fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(C.TASKS, color=C.INK2, fontsize=9)
        ax.set_ylabel("return", color=C.INK2, fontsize=9)
        ax.set_ylim(0, max(pm) * 1.25)

    h = [Line2D([0], [0], color=C.PAPER_COLOR, ls="--", lw=1.4), Patch(color="#a8a8a8")] + [Line2D([0], [0], color=c["color"], lw=2) for c, _ in conds]
    lb = ["論文(10 seed 平均、帯・誤差棒=±std)", "論文(棒)"] + [f"{c['label']}・n={C.n_seeds(d)} seed" for c, d in conds]
    fig.legend(h, lb, loc="upper left", bbox_to_anchor=(0.05, 0.955), ncol=2, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.suptitle("MAGPO @ CoordSum: 4条件 vs 論文", x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold", color=C.INK)
    fig.subplots_adjust(top=0.82, bottom=0.07, left=0.06, right=0.98)
    fig.savefig(out, dpi=150, facecolor=C.SURF)
    print("saved", out)


def fig_final_all(paper, out):
    """全手法の最終スコア。薄い色=コード既定値、濃い色=論文既定値(MAGPO は旧版コード、CTDS は旧版コードベースでコード既定値のみ)。"""
    data = {k: load_cond(k) for k in {c for p in PAIRS.values() for c in p if c}}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), facecolor=C.SURF)
    w = 0.26
    for ax, t in zip(axes, C.TASKS):
        C.style_ax(ax)
        for i, s in enumerate(FIG_SYSTEMS):
            m, sd = C.paper_stats(paper, s, t, "final")
            ax.bar(i - w, m, w * 0.9, yerr=sd, color="#a8a8a8", ecolor=C.INK2, error_kw=dict(lw=1, capsize=2))
            for j, (ck, al) in enumerate(zip(PAIRS[s], (0.45, 1.0))):
                if ck:
                    mean_bar(ax, i + j * w, C.seed_metrics(data[ck], s, t, "final"), w, C.COLOR[s], alpha=al)
        ax.set_xticks(range(len(FIG_SYSTEMS))); ax.set_xticklabels([C.LABEL[s] for s in FIG_SYSTEMS], fontsize=8, color=C.INK2)
        ax.set_title(f"CoordSum-{t}", loc="left", fontsize=10, fontweight="bold", color=C.INK)
        ax.set_ylim(0, 185)
    axes[0].set_ylabel(f"最終スコア(最後の{C.LAST}評価点の平均)", fontsize=9, color=C.INK2)
    n_code = min(n_of(data[PAIRS[s][0]], [s]) for s in FIG_SYSTEMS)
    n_pd = min(n_of(data[PAIRS[s][1]], [s]) for s in FIG_SYSTEMS if PAIRS[s][1])
    h = [Patch(color="#a8a8a8"), Patch(color=C.INK2, alpha=0.45), Patch(color=C.INK2)]
    fig.legend(h, ["論文(10 seed 平均±std)",
                   f"自分: コード既定値(MAGPO・CTDS は旧版コードベース)n={n_code} seed",
                   f"自分: 論文既定値(MAGPO は旧版コード。CTDS は実行なし)n={n_pd} seed"],
               loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=140, facecolor=C.SURF)
    print("saved", out)


def fig_curves(paper, out):
    """手法×タスクの学習曲線。薄い線=コード既定値、濃い線=論文既定値(MAGPO・他は現行コード。CTDS は旧版コードベースでコード既定値のみ)。"""
    data = {k: load_cond(k) for k in {c for p in CURRENT_PAIRS.values() for c in p if c}}
    nsys = len(FIG_SYSTEMS)
    fig, axes = plt.subplots(len(C.TASKS), nsys, figsize=(3.4 * nsys, 2.6 * len(C.TASKS)), sharex=True, facecolor=C.SURF)
    for i, t in enumerate(C.TASKS):
        ymax = max(paper_runs(paper, s, t)[1].max() for s in FIG_SYSTEMS) * 1.05
        for j, s in enumerate(FIG_SYSTEMS):
            ax = axes[i][j]; C.style_ax(ax)
            st, arr = paper_runs(paper, s, t)
            m, sd = arr.mean(0), arr.std(0, ddof=1)
            ax.fill_between(st / 1e7, m - sd, m + sd, color=C.COLOR[s], alpha=0.18, lw=0)
            ax.plot(st / 1e7, m, color=C.COLOR[s], ls="--", lw=1.2)
            for ck, al, lw in zip(CURRENT_PAIRS[s], (0.4, 1.0), (1.2, 1.8)):
                if not ck:
                    continue
                sk = C.stack(data[ck], s, t)
                if sk:
                    C.band(ax, sk[0], sk[1], C.COLOR[s], lw=lw, alpha=al)
            ax.set_ylim(0, ymax)
            if i == 0:
                ax.set_title(C.LABEL[s], loc="left", fontsize=10, fontweight="bold", color=C.INK)
            if j == 0:
                ax.set_ylabel(f"{t}\nreturn", fontsize=8, color=C.INK2)
            if i == len(C.TASKS) - 1:
                ax.set_xlabel("timesteps (1e7)", fontsize=8, color=C.INK2)
    n_code = min(n_of(data[CURRENT_PAIRS[s][0]], [s]) for s in FIG_SYSTEMS if s != "magpo")
    n_magpo = n_of(data[CURRENT_PAIRS["magpo"][0]], ["magpo"])
    n_pd = min(n_of(data[CURRENT_PAIRS[s][1]], [s]) for s in FIG_SYSTEMS if CURRENT_PAIRS[s][1])
    h = [Line2D([0], [0], color=C.INK2, ls="--", lw=1.2), Line2D([0], [0], color=C.INK2, alpha=0.4, lw=1.2), Line2D([0], [0], color=C.INK2, lw=1.8)]
    fig.legend(h, ["論文: 10 seed 平均(帯 = ±std)",
                   f"自分: コード既定値 GAE0.95/vf0.5(n={n_code} seed。MAGPO のみ n={n_magpo}。複数 seed は平均 ± std)",
                   f"自分: 論文既定値 GAE0.9/vf0.1(n={n_pd} seed。CTDS は実行なし)"],
               loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.text(0.5, 0.945, "MAGPO・他手法は現行コード(MAGPO も学習側 RNN の隠れ状態を更新する最新版)。CTDS のみ旧版コードベース", ha="center", fontsize=8, color=C.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor=C.SURF)
    print("saved", out)


def fig_repro_curves(paper, out):
    """05: 採用条件(MAGPO=旧版コード、全手法=コード既定値)の学習曲線を論文と比べる。"""
    data = {k: load_cond(k) for k in set(REPRO.values())}
    fig, axes = plt.subplots(len(C.TASKS), len(FIG_SYSTEMS), figsize=(3.4 * len(FIG_SYSTEMS), 2.6 * len(C.TASKS)), sharex=True, facecolor=C.SURF)
    for i, t in enumerate(C.TASKS):
        ymax = max(paper_runs(paper, s, t)[1].max() for s in FIG_SYSTEMS) * 1.05
        for j, s in enumerate(FIG_SYSTEMS):
            ax = axes[i][j]; C.style_ax(ax)
            st, arr = paper_runs(paper, s, t)
            m, sd = arr.mean(0), arr.std(0, ddof=1)
            ax.fill_between(st / 1e7, m - sd, m + sd, color="#888888", alpha=0.2, lw=0)
            ax.plot(st / 1e7, m, color="#555555", ls="--", lw=1.3)
            sk = C.stack(data[REPRO[s]], s, t)
            if sk:
                C.band(ax, sk[0], sk[1], C.COLOR[s], lw=1.9, alpha=1.0)
            ax.set_ylim(0, ymax)
            if i == 0:
                ax.set_title(C.LABEL[s], loc="left", fontsize=10, fontweight="bold", color=C.INK)
            if j == 0:
                ax.set_ylabel(f"{t}\nreturn", fontsize=8, color=C.INK2)
            if i == len(C.TASKS) - 1:
                ax.set_xlabel("timesteps (1e7)", fontsize=8, color=C.INK2)
    n_all = min(n_of(data[REPRO[s]], [s]) for s in FIG_SYSTEMS)
    h = [Line2D([0], [0], color="#555555", ls="--", lw=1.3), Line2D([0], [0], color=C.INK2, lw=1.9)]
    fig.legend(h, ["論文: 10 seed 平均(帯 = ±std)", f"自分: 平均 ± std(色=手法、n={n_all} seed)"], loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.text(0.5, 0.945, "MAGPO・CTDS=旧版コード(隠れ状態を更新しない版)、他=現行コード。全手法コード既定値(GAE 0.95 / vf 0.5)", ha="center", fontsize=8, color=C.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor=C.SURF)
    print("saved", out)


def fig_repro_final(paper, out):
    """06: 同条件の最終スコア(棒=自分の seed 平均、点=各 seed、灰=論文平均±std)。"""
    data = {k: load_cond(k) for k in set(REPRO.values())}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), facecolor=C.SURF)
    w = 0.38
    for ax, t in zip(axes, C.TASKS):
        C.style_ax(ax)
        for i, s in enumerate(FIG_SYSTEMS):
            m, sd = C.paper_stats(paper, s, t, "final")
            ax.bar(i - w / 2, m, w * 0.92, yerr=sd, color="#a8a8a8", ecolor=C.INK2, error_kw=dict(lw=1, capsize=2))
            v = C.seed_metrics(data[REPRO[s]], s, t, "final")
            if v is None:
                continue
            ax.bar(i + w / 2, v.mean(), w * 0.92, color=C.COLOR[s], yerr=v.std(ddof=1) if len(v) > 1 else None,
                   ecolor=C.INK2, error_kw=dict(lw=1, capsize=2))
            ax.scatter([i + w / 2] * len(v), v, s=10, color=C.INK, zorder=3)
            ref = np.array([C.metric(y, "final") for k, (_, y) in paper.items() if k[0] == s and k[1] == t])
            ax.text(i, 168, f"{v.mean():.0f} (n={len(v)})\nz={C.z_score(v, ref):+.1f}", ha="center", fontsize=7.5, color=C.INK2)
        ax.set_xticks(range(len(FIG_SYSTEMS))); ax.set_xticklabels([C.LABEL[s] for s in FIG_SYSTEMS], fontsize=8, color=C.INK2)
        ax.set_title(f"CoordSum-{t}", loc="left", fontsize=10, fontweight="bold", color=C.INK)
        ax.set_ylim(0, 185)
    axes[0].set_ylabel(f"最終スコア(最後の{C.LAST}評価点の平均)", fontsize=9, color=C.INK2)
    h = [Patch(color="#a8a8a8"), Patch(color=C.INK2)]
    fig.legend(h, ["論文(10 seed 平均±std)", "自分(MAGPO・CTDS=旧版コード、他=現行コード。コード既定値。点=各 seed。上段: 平均 (n=seed 数) と論文との z(Welch 型))"],
               loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=140, facecolor=C.SURF)
    print("saved", out)

def fig_normalized(ours, set_name, out):
    """論文 Figure (a) 形式。正規化は plot_coordsum.py と同じ関数を使う(タスクごとの最大値で割る)。"""
    import plot_coordsum as pc  # 同じ analysis/ にある

    paper = pc.load(C.PAPER_CSV)
    params = pc.norm_params(paper, "maxmean")
    s = C.SETS[set_name]
    n = min(C.n_seeds({k: v for k, v in ours.items() if k[0] == m}) for m in FIG_SYSTEMS if any(k[0] == m for k in ours))
    ctds_note = "・CTDS=旧版ベース" if any(k[0] == "ctds" for k in ours) else ""
    titles = {"paper": "論文(10 seed)", "ours": f"自分(n={n} seed): MAGPO={C.COND[s['magpo']]['short']}{ctds_note}"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, facecolor=C.SURF)
    for ax, (name, data) in zip(axes, (("paper", paper), ("ours", ours))):
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
    ap.add_argument("--set", default="codedefault", choices=list(C.SETS))
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", default="1,2,3,4,5,6", help="作る図の番号(カンマ区切り)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    want = set(a.only.split(","))
    paper = C.load_paper()
    ours = C.load_set(a.set) if "4" in want else None
    if ours is not None and a.set == "codedefault":  # CTDS はコード既定値の実行のみ
        ours.update(load_cond(CTDS_KEY))
    if "1" in want:
        fig_conditions(paper, os.path.join(a.outdir, "01_magpo_conditions.png"))
    if "2" in want:
        fig_final_all(paper, os.path.join(a.outdir, "02_final_vs_paper.png"))
    if "3" in want:
        fig_curves(paper, os.path.join(a.outdir, "03_curves_by_task.png"))
    if "5" in want:
        fig_repro_curves(paper, os.path.join(a.outdir, "05_repro_curves.png"))
    if "6" in want:
        fig_repro_final(paper, os.path.join(a.outdir, "06_repro_final.png"))
    if "4" in want:
        fig_normalized(ours, a.set, os.path.join(a.outdir, "04_normalized_overview.png"))


if __name__ == "__main__":
    main()
