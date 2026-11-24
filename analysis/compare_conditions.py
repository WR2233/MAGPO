"""実験条件ごとの CoordSum の結果を、論文の分布と比べる(表を標準出力へ)。

指標(生の return。1 run ごとに計算し、論文は 10 seed の分布を作る):
  final : 最後の 10 評価点の平均(最終付近。1 点だけより評価ノイズに強い)
  auc   : 全評価点の平均(学習の速さ。立ち上がりが遅いと小さい)
z: 自分が 1 seed なら (自分 - 論文平均) / 論文の seed 間 std。複数 seed なら、平均の差を
   sqrt(論文var/10 + 自分var/n) で割った Welch 型(common.z_score)。
未完了の run(評価点が 122 に満たない)は除外する。

使い方(条件は analysis/common.py の登録簿を既定で全部使う):
  uv run --no-project --with numpy python analysis/compare_conditions.py
  # 条件を指定する場合: --cond "ラベル=csvパス" を繰り返す。
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

KINDS = (("final", f"最終付近(最後の{C.LAST}評価点の平均)"), ("auc", "学習の速さ(全評価点の平均)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=C.PAPER_CSV)
    ap.add_argument("--cond", action="append", help='"ラベル=csvパス"(省略時は登録簿の全条件)')
    a = ap.parse_args()

    paper = C.load_runs(a.paper)
    cond_args = a.cond or [f"{c['short']}={os.path.join(C.DATA, c['csv'])}" for c in C.CONDITIONS]
    conds = {}
    for c in cond_args:
        label, path = c.split("=", 1)
        conds[label] = C.load_runs(path)
    labels = list(conds)

    summary = defaultdict(list)  # (kind, system, label) -> z の列
    for kind, title in KINDS:
        print(f"\n{'=' * 110}\n指標: {title}   z = 論文との差(標準偏差単位。複数 seed は Welch 型)。セル: 平均 (z, n=seed 数)\n{'=' * 110}")
        print(f"{'手法':7s} {'タスク':9s} {'論文 mean±std':>16s} | " + " | ".join(f"{l[:14]:>26s}" for l in labels))
        for s in C.SYSTEMS:
            for t in C.TASKS:
                ref = np.array([C.metric(y, kind) for k, (_, y) in paper.items() if k[0] == s and k[1] == t])
                if len(ref) < 2:
                    continue
                cells = []
                for l in labels:
                    v = C.seed_metrics(conds[l], s, t, kind)
                    if v is None:
                        cells.append(f"{'—':>26s}")
                        continue
                    z = C.z_score(v, ref)
                    summary[(kind, s, l)].append(z)
                    cells.append(f"{v.mean():8.1f} (z={z:+5.1f}, n={len(v)})".rjust(26))
                print(f"{s:7s} {t:9s} {ref.mean():8.1f}±{ref.std(ddof=1):5.1f} | " + " | ".join(cells))

    print(f"\n{'=' * 110}\n手法ごとの要約: 4 タスクの |z| の平均(小さいほど論文に近い) / z の平均(符号=高低の傾向)\n{'=' * 110}")
    for kind, _ in KINDS:
        print(f"[{kind}]")
        for s in C.SYSTEMS:
            row = []
            for l in labels:
                zs = summary.get((kind, s, l), [])
                row.append(f"{np.mean(np.abs(zs)):5.2f} / {np.mean(zs):+5.2f}" if len(zs) == len(C.TASKS) else "    —    ")
            print(f"  {s:7s} " + " | ".join(f"{l[:12]}: {r}" for l, r in zip(labels, row)))


if __name__ == "__main__":
    main()
