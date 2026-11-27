"""実行ログ(起動時に出力される実際の config)から、各 run が実際に使ったハイパラを読み取り、
analysis/data/ours_params.csv(予定値)と突き合わせる。
使い方: python3 analysis/verify_params_from_logs.py <logs_root> [--out analysis/data/ours_params_actual.csv] [--partial]
  --partial : 実行中(評価 122 回に満たない)のログも対象にする(設定の確認用)
  logs_root の下に ディレクトリ名=MAGPO / MAGPO_paperdef / MAGPO_old / MAGPO_oldpd / CTDS(各 logs の中身)を置く
  ログ先頭の「== code: commit=...」行があれば、commit 番号も出力 CSV に載せる(code_commit / code_dirty)
"""
import ast, csv, glob, os, re, sys

COND = {"MAGPO": "v1_codedefault", "MAGPO_paperdef": "v2_paperdefault",
        "MAGPO_old": "old_codedefault", "MAGPO_oldpd": "old_paperdefault",
        "CTDS": "ctds_codedefault"}
JOB_COND = {"10422803": "ctds_klall"}  # CTDS 前版(guider に KL を常にかける)のジョブ。他の CTDS ジョブは ctds_codedefault
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TASK_CSV = {"3x10-30": "3x10", "3x30-50": "3x30", "5x20-80": "5x20", "8x15-100": "8x15"}


def parse_config(text):
    """rich の pretty print から {'section.sub.key': value} を作る(最初の config 出力のみ)。"""
    flat, stack = {}, []
    started = False
    for line in text.splitlines():
        m = re.match(r"^((?:│   )*)'?([\w_]+)'?: (.*?),?$", line.rstrip()) if "│" in line else None
        if not m:
            continue
        depth = len(m.group(1)) // 4
        key, val = m.group(2), m.group(3)
        if key == "logger" and depth == 1:
            started = True
        if not started:
            continue
        stack = stack[: depth - 1]
        if val == "{":
            stack.append(key)
            continue
        try:
            v = ast.literal_eval(val)
        except Exception:
            v = val
        flat[".".join(stack + [key])] = v
        if key == "num_updates_per_eval" and "system" in stack:
            break
    return flat


def main():
    root = sys.argv[1]
    partial = "--partial" in sys.argv
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "analysis/data/ours_params_actual.csv"
    exp = list(csv.DictReader(open("analysis/data/ours_params.csv")))
    rows, issues = [], []
    for d, cond0 in COND.items():
        for f in sorted(glob.glob(f"{root}/logs_{d}/coordsum_*_s*_*.out")):
            text = ANSI.sub("", open(f, errors="ignore").read())
            n_eval = text.count("MISC - Timestep")
            if n_eval < 122 and not partial:
                continue  # 未完了/テスト/失敗は対象外
            m = re.match(r".*coordsum_(\w+?)_(\d+x\d+-\d+)_s(\d+)_(\d+)\.out", f)
            system, task, seed, job = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            cfg = parse_config(text)
            if not cfg:
                issues.append(f"[config読取失敗] {f}")
                continue
            cond = JOB_COND.get(job, cond0) if d == "CTDS" else cond0
            mc = re.search(r"== code: commit=(\w+)(?: branch=(\S+))?(?: dirty=(\w+))?", text)
            cfg.update(_commit=mc.group(1) if mc else "", _dirty=(mc.group(3) or "") if mc else "", _cond=cond, _system=system, _task=task, _seed=seed, _job=job, _log=os.path.basename(f))
            rows.append(cfg)
            if cfg.get("system.seed") != seed:
                issues.append(f"[不一致] {cond} {system} {task} job{job}: seed ファイル名={seed} 実際={cfg.get('system.seed')}")
            e = next((r for r in exp if r["condition"] == cond and r["system_name"] == system and r["task"] == TASK_CSV[task]), None)
            if e is None:
                issues.append(f"[予定行なし] {cond} {system} {task}")
                continue
            lr_exp = e["actor_lr"] or e["critic_lr"]
            chk = {  # 予定列 -> (実際のキー候補)
                "num_envs": ["arch.num_envs"], "num_updates": ["system.num_updates"],
                "num_evaluation": ["arch.num_evaluation"], "ppo_epochs": ["system.ppo_epochs"],
                "num_minibatches": ["system.num_minibatches"], "clip_eps": ["system.clip_eps"],
                "ent_coef": ["system.ent_coef"], "max_grad_norm": ["system.max_grad_norm"],
                "alpha": ["system.alpha"], "delta": ["system.clip_gpo"],
                "gae_lambda": ["system.gae_lambda"], "vf_coef": ["system.vf_coef"],
                "n_head": ["network.net_config.n_head", "network.n_head"],
                "n_block": ["network.net_config.n_block", "network.n_block"],
                "n_embd": ["network.net_config.embed_dim", "network.embed_dim"],
                "decay_scaling_factor": ["network.memory_config.decay_scaling_factor"],
                "recurrent_chunk_size": ["system.recurrent_chunk_size"],
            }
            for col, keys in chk.items():
                ev = (e[col] or "").strip()
                av = next((cfg[k] for k in keys if k in cfg), None)
                if ev == "" or av is None:
                    continue
                if abs(float(ev) - float(av)) > 1e-12 * max(1, abs(float(ev))):
                    issues.append(f"[不一致] {cond} {system} {task} job{job}: {col} 予定={ev} 実際={av}")
            alr = cfg.get("system.actor_lr")
            if alr is not None and lr_exp and abs(float(lr_exp) - float(alr)) > 1e-12:
                issues.append(f"[不一致] {cond} {system} {task} job{job}: actor_lr 予定={lr_exp} 実際={alr}")
    keys = sorted({k for r in rows for k in r if not k.startswith("_")})
    keep = [k for k in keys if k.startswith(("system.", "arch.num", "network.net_config", "network.embed_dim", "network.n_", "network.memory_config.decay", "env.scenario", "env.name", "env.task"))]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "system", "task", "seed", "job", "log", "code_commit", "code_dirty"] + keep, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"condition": r["_cond"], "system": r["_system"], "task": r["_task"], "seed": r["_seed"], "job": r["_job"], "log": r["_log"], "code_commit": r["_commit"], "code_dirty": r["_dirty"], **{k: r.get(k) for k in keep}})
    print(f"読取 {len(rows)} run → {out}")
    from collections import Counter
    print(Counter((r["_cond"], r["_system"]) for r in rows))
    print("\n".join(issues) if issues else "不一致なし")


if __name__ == "__main__":
    main()
