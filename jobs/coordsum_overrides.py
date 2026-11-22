"""experiment_data/params.csv から、CoordSum 用の Hydra override と実行ファイルを出力する。

使い方:  python jobs/coordsum_overrides.py SYSTEM TASK
  SYSTEM: magpo | mappo | happo | sable | mat
  TASK  : 3x10-30 | 3x30-50 | 5x20-80 | 8x15-100
出力:   1行目に実行する python ファイル、2行目以降に1行1個の override。
標準ライブラリのみ使用(計算ノードの .venv でも動く)。

対応づけの仮定(params.csv の列名 -> コードのキー):
  - magpo の delta -> system.clip_gpo
  - magpo / sable / mat は学習率が1つだけ(system.actor_lr)。CoordSum の行では
    その値が critic_lr 列に入っているため、actor_lr が空なら critic_lr を使う
  - mappo / happo は actor_lr と critic_lr の両方を設定する
  - 論文 付録 Table 2 / Table 3 の既定値: GAE=0.9, value function coefficient=0.1 を全手法に適用する
    (リポジトリの設定ファイルの既定値は 0.95 / 0.5 で、params.csv にもこの2項目は無い)。
    環境変数 PAPER_DEFAULTS=0 で無効化できる(以前の実行との比較用)
"""
import csv
import os
import sys

SCRIPTS = {
    "magpo": "mava/systems/gpo/anakin/rec_magpo.py",
    "mappo": "mava/systems/ppo/anakin/rec_mappo.py",
    "happo": "mava/systems/ppo/anakin/rec_happo.py",
    "sable": "mava/systems/sable/anakin/rec_sable.py",
    "mat": "mava/systems/mat/anakin/mat.py",
}
TASK_TO_CSV = {"3x10-30": "3x10", "3x30-50": "3x30", "5x20-80": "5x20", "8x15-100": "8x15"}
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment_data", "params.csv")


def load_row(system, task):
    rows = [
        r for r in csv.DictReader(open(CSV_PATH))
        if r["system_name"] == system and r["env_name"] == "coordsum" and r["task"] == TASK_TO_CSV[task]
    ]
    if len(rows) != 1:
        sys.exit(f"params.csv: {system}/{task} に一致する行が {len(rows)} 件")
    return {k: (v or "").strip() for k, v in rows[0].items()}


def build(system, task):
    r = load_row(system, task)
    out = []

    def add(key, col):
        if r.get(col):
            out.append(f"{key}={r[col]}")

    if os.environ.get("PAPER_DEFAULTS", "1") == "1":
        out += ["system.gae_lambda=0.9", "system.vf_coef=0.1"]
    add("arch.num_envs", "num_envs")
    add("system.num_updates", "num_updates")
    add("arch.num_evaluation", "num_evaluation")
    for k in ("ppo_epochs", "num_minibatches", "clip_eps", "ent_coef", "max_grad_norm"):
        add(f"system.{k}", k)

    if system in ("mappo", "happo"):
        add("system.actor_lr", "actor_lr")
        add("system.critic_lr", "critic_lr")
        add("system.recurrent_chunk_size", "recurrent_chunk_size")
    else:
        lr = r.get("actor_lr") or r.get("critic_lr")
        if lr:
            out.append(f"system.actor_lr={lr}")

    if system == "magpo":
        add("system.alpha", "alpha")
        add("system.clip_gpo", "delta")
    if system in ("magpo", "sable"):
        add("network.net_config.embed_dim", "n_embd")
        add("network.net_config.n_block", "n_block")
        add("network.net_config.n_head", "n_head")
        add("network.memory_config.decay_scaling_factor", "decay_scaling_factor")
    if system == "mat":
        add("network.embed_dim", "n_embd")
        add("network.n_block", "n_block")
        add("network.n_head", "n_head")
    return SCRIPTS[system], out


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in SCRIPTS or sys.argv[2] not in TASK_TO_CSV:
        sys.exit(__doc__)
    script, overrides = build(sys.argv[1], sys.argv[2])
    print(script)
    print("\n".join(overrides))
