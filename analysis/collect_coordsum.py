"""CoordSum の学習曲線を「生の return のまま」縦持ち CSV に集める(正規化はしない)。

列: source(ours|paper), system, task, seed, step_count, mean_episode_return
正規化は plot_coordsum.py 側で行う。ここでは生データだけを残し、後で正規化の定義を
変えても再収集せずに描き直せるようにする。

使い方:
  # 論文の生データ(experiment_data.json)から CoordSum だけ抜き出す
  python analysis/collect_coordsum.py paper experiment_data/experiment_data.json analysis/data/paper_coordsum.csv

  # 自分の実行結果(results/coordsum_<system>_<task>_s<seed>/json/**/metrics.json)を集める
  python analysis/collect_coordsum.py ours results analysis/data/ours_coordsum.csv

標準ライブラリのみ使用。
"""
import csv
import glob
import json
import os
import re
import sys

FIELDS = ["source", "system", "task", "seed", "step_count", "mean_episode_return"]
STEP = re.compile(r"^step_(\d+)$")


def mean(x):
    x = x if isinstance(x, list) else [x]
    return sum(x) / len(x)


def rows_from_seed(source, system, task, seed, seed_dict):
    for k, v in seed_dict.items():
        if STEP.match(k):
            yield [source, system, task, seed, int(v["step_count"]), mean(v["mean_episode_return"])]


def collect_paper(path):
    data = json.load(open(path))["CoordSum"]
    for scen, by_sys in data.items():
        task = scen.replace("CoordSum-", "")
        for system, by_seed in by_sys.items():
            for sk, sd in by_seed.items():
                if sk.startswith("seed_"):
                    yield from rows_from_seed("paper", system, task, int(sk.split("_")[1]), sd)


def collect_ours(root):
    pat = re.compile(r"coordsum_(?P<system>[a-z]+)_(?P<task>\d+x\d+-\d+)_s(?P<seed>\d+)")
    for p in sorted(glob.glob(os.path.join(root, "coordsum_*", "json", "**", "metrics.json"), recursive=True)):
        m = pat.search(p)
        if not m:
            continue
        d = json.load(open(p))
        # {env: {task: {algo: {seed_N: {...}}}}} を辿り、seed 辞書を取り出す
        env = next(iter(d.values()))
        scen = next(iter(env.values()))
        algo = next(iter(scen.values()))
        for sk, sd in algo.items():
            if sk.startswith("seed_"):
                yield from rows_from_seed("ours", m["system"], m["task"], int(m["seed"]), sd)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("paper", "ours"):
        sys.exit(__doc__)
    kind, src, out = sys.argv[1:]
    rows = list(collect_paper(src) if kind == "paper" else collect_ours(src))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    systems = sorted({r[1] for r in rows})
    tasks = sorted({r[2] for r in rows})
    print(f"wrote {out}: {len(rows)} rows | systems={systems} | tasks={tasks}")


if __name__ == "__main__":
    main()
