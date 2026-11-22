"""MAGPO の学習ログから学習曲線を描く。

使い方:
    uv run --no-project --with matplotlib --with numpy \
        python analysis/plot_training.py LOG.out OUT.png [--title "..."]

LOG.out は rec_magpo.py のコンソール出力(ANSI カラーコード付きでよい)。
"""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", "DejaVu Sans"]

# 検証済みパレット(light): series-1 = blue, series-2 = orange。surface = #fcfcfb
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUM = r"(-?\d+(?:\.\d+)?)"


def parse(path):
    rows, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = ANSI.sub("", line).strip()
        if line.startswith("MISC - Timestep:"):
            cur = {"step": int(line.split(":")[1])}
            rows.append(cur)
        elif cur is not None and line.startswith("ACTOR"):
            cur["a_ret"] = float(re.search(rf"Episode return mean: {NUM}", line).group(1))
            cur["a_len"] = float(re.search(rf"Episode length mean: {NUM}", line).group(1))
        elif cur is not None and line.startswith("EVALUATOR"):
            cur["e_ret"] = float(re.search(rf"Episode return mean: {NUM}", line).group(1))
            cur["e_len"] = float(re.search(rf"Episode length mean: {NUM}", line).group(1))
        elif cur is not None and line.startswith("TRAINER"):
            cur["ent"] = float(re.search(rf"Entropy: {NUM}", line).group(1))
    keys = ["step", "a_ret", "a_len", "e_ret", "e_len", "ent"]
    return {k: np.array([r[k] for r in rows if all(x in r for x in keys)]) for k in keys}


def style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")


def direct_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points",
                color=INK, fontsize=9, va="center")
    ax.plot(x[-1], y[-1], "o", ms=6, color=color, mec=SURFACE, mew=2, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--title", default="MAGPO on LBF 2s-8x8-2p-2f-coop")
    a = ap.parse_args()

    d = parse(a.log)
    x = d["step"] / 1e6

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, facecolor=SURFACE)

    ax = axes[0]
    style(ax, "episode return", "Return: 約1Mステップで最大値 1.0 に到達(青は橙の下に重なる)")
    ax.plot(x, d["a_ret"], color=BLUE, lw=2)
    ax.plot(x, d["e_ret"], color=ORANGE, lw=2)
    ax.set_ylim(-0.02, 1.05)
    direct_label(ax, x, d["a_ret"], "学習時 (ACTOR)", BLUE, dy=7)
    direct_label(ax, x, d["e_ret"], "評価 (EVALUATOR)", ORANGE, dy=-7)

    ax = axes[1]
    style(ax, "episode length (steps)", "Episode length: 少ないほど効率的(上限 100)")
    ax.plot(x, d["a_len"], color=BLUE, lw=2)
    ax.plot(x, d["e_len"], color=ORANGE, lw=2)
    ax.set_ylim(0, 105)
    direct_label(ax, x, d["a_len"], "学習時", BLUE, dy=7)
    direct_label(ax, x, d["e_len"], "評価", ORANGE, dy=-7)

    ax = axes[2]
    style(ax, "entropy", "Entropy: 方策のランダムさ")
    ax.plot(x, d["ent"], color=BLUE, lw=2)
    ax.set_ylim(0, None)
    ax.set_xlabel("environment steps (millions)", color=INK2, fontsize=9)

    for ax in axes:
        ax.set_xlim(0, x[-1])
        ax.margins(x=0)
    fig.suptitle(a.title, x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    fig.subplots_adjust(right=0.82)
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print("saved", a.out, "| points:", len(x), "| final:",
          f"actor ret {d['a_ret'][-1]:.3f}, eval ret {d['e_ret'][-1]:.3f}, len {d['e_len'][-1]:.1f}")


if __name__ == "__main__":
    main()
