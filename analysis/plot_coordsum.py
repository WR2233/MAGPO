"""論文 Figure (a) 相当: CoordSum の学習曲線(4タスク平均の正規化 return)を描く。

入力は collect_coordsum.py が作った「正規化前の」CSV。正規化はここで選ぶ。

使い方:
  uv run --no-project --with matplotlib --with numpy python analysis/plot_coordsum.py \
      --paper analysis/data/paper_coordsum.csv --ours analysis/data/ours_coordsum.csv \
      --norm maxmean --out figures/coordsum_fig_a.png

--norm(タスクごとの正規化。定数は基準データ --norm-ref の分から計算し、ours / paper の両方へ同じ定数を適用する):
  none     : 正規化しない(生の return)
  maxmean  : 定数 = 全手法の「seed 平均曲線」の最大値            -> x / 定数
  maxabs   : 定数 = 全手法・全seed・全ステップの最大値           -> x / 定数
  minmax   : (x - 最小値) / (maxmean の定数 - 最小値)。最小値は基準データ全体の最小

論文の正規化の定義は未確認。定義が分かったら --norm を足すだけで、生データを再収集せず描き直せる。
"""
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# 検証済みパレットの先頭6スロット(固定順)。色は「手法」に固定する。
ORDER = ["magpo", "ctds", "mappo", "happo", "mat", "sable"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
LSTYLE = dict(zip(ORDER, ["-", "-", "--", "-.", ":", (0, (5, 1, 1, 1))]))  # 色以外でも区別する
LABEL = {"magpo": "MAGPO", "ctds": "CTDS", "mappo": "MAPPO", "happo": "HAPPO", "mat": "MAT", "sable": "SABLE"}


def load(path):
    """-> {(system, task, seed): (steps[T], returns[T])}"""
    tmp = defaultdict(list)
    for r in csv.DictReader(open(path)):
        tmp[(r["system"], r["task"], int(r["seed"]))].append((int(r["step_count"]), float(r["mean_episode_return"])))
    out = {}
    for k, v in tmp.items():
        v.sort()
        out[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v]))
    return out


def norm_params(ref, mode):
    """タスクごとの (最小値, 定数) を基準データ ref から計算する。"""
    ref = {k: v for k, v in ref.items() if k[0] in ORDER}  # 図に出る手法だけで定数を決める(ippo は除外)
    tasks = sorted({k[1] for k in ref})
    p = {}
    for t in tasks:
        allv = np.concatenate([v[1] for k, v in ref.items() if k[1] == t])
        mean_curves = []
        for s in sorted({k[0] for k in ref if k[1] == t}):
            seeds = [v[1] for k, v in ref.items() if k[0] == s and k[1] == t]
            n = min(len(x) for x in seeds)
            mean_curves.append(np.mean([x[:n] for x in seeds], axis=0))
        maxmean = max(c.max() for c in mean_curves)
        lo = 0.0
        if mode == "none":
            scale = 1.0
        elif mode == "maxmean":
            scale = maxmean
        elif mode == "maxabs":
            scale = allv.max()
        elif mode == "minmax":
            lo = allv.min()
            scale = maxmean - lo
        else:
            raise SystemExit(f"unknown --norm {mode}")
        p[t] = (lo, scale)
    return p


def curves(data, params):
    """手法ごとに、seed ごとの「タスク平均の正規化 return」曲線 -> {system: (steps, [seed, T])}"""
    by_sys = defaultdict(lambda: defaultdict(dict))  # system -> seed -> task -> (steps, y)
    for (s, t, sd), (st, y) in data.items():
        lo, sc = params[t]
        by_sys[s][sd][t] = (st, (y - lo) / sc)
    out = {}
    for s, by_seed in by_sys.items():
        rows, steps = [], None
        for sd, by_task in sorted(by_seed.items()):
            n = min(len(v[1]) for v in by_task.values())
            rows.append(np.mean([v[1][:n] for v in by_task.values()], axis=0))
            steps = next(iter(by_task.values()))[0][:n]
        n = min(len(r) for r in rows)
        out[s] = (steps[:n], np.array([r[:n] for r in rows]), sorted({t for sd in by_seed.values() for t in sd}))
    return out


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel("timesteps (1e7)", color=INK2, fontsize=9)


def draw(ax, cv):
    for s in ORDER:
        if s not in cv:
            continue
        steps, arr, _ = cv[s]
        m = arr.mean(0)
        x = steps / 1e7
        ax.plot(x, m, color=COLOR[s], ls=LSTYLE[s], lw=2, label=LABEL[s])
        if arr.shape[0] > 1:
            se = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
            ax.fill_between(x, m - se, m + se, color=COLOR[s], alpha=0.18, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper")
    ap.add_argument("--ours")
    ap.add_argument("--norm", default="maxmean", choices=["none", "maxmean", "maxabs", "minmax"])
    ap.add_argument("--norm-ref", default="paper", choices=["paper", "ours"])
    ap.add_argument("--out", default="figures/coordsum_fig_a.png")
    a = ap.parse_args()

    sets = {}
    if a.ours:
        sets["ours"] = load(a.ours)
    if a.paper:
        sets["paper"] = load(a.paper)
    ref = sets[a.norm_ref] if a.norm_ref in sets else next(iter(sets.values()))
    params = norm_params(ref, a.norm)
    print(f"norm={a.norm} ref={a.norm_ref if a.norm_ref in sets else 'fallback'} (task: min, scale)")
    for t, (lo, sc) in params.items():
        print(f"  {t}: min={lo:.3f} scale={sc:.3f}")

    titles = {"ours": "再現(自分の実行)", "paper": "論文の生データ"}
    fig, axes = plt.subplots(1, len(sets), figsize=(max(7.8, 5.6 * len(sets)), 4.6), sharey=True, facecolor=SURFACE, squeeze=False)
    rows = []
    for ax, (name, data) in zip(axes[0], sets.items()):
        cv = curves(data, params)
        style(ax, titles[name])
        draw(ax, cv)
        for s in ORDER:
            if s in cv:
                steps, arr, tasks = cv[s]
                rows.append((name, LABEL[s], arr.shape[0], len(tasks), float(arr[:, -1].mean())))
    axes[0][0].set_ylabel("Mean episode return" + ("" if a.norm == "none" else f" (normalized: {a.norm})"), color=INK2, fontsize=9)
    h, l = axes[0][-1].get_legend_handles_labels()
    if not h:
        h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(h), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print(f"saved {a.out}\n\n最終ステップの値(表: 色が薄い系列の読み取り用)")
    print(f"{'source':6s} {'system':7s} {'seeds':>5s} {'tasks':>5s} {'final':>8s}")
    for r in rows:
        print(f"{r[0]:6s} {r[1]:7s} {r[2]:5d} {r[3]:5d} {r[4]:8.3f}")


if __name__ == "__main__":
    main()
