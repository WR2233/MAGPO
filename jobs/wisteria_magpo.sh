#!/bin/bash
# Wisteria/BDEC-01 (Aquarius, A100) 用 PJM ジョブスクリプトの雛形。
# 記憶ベースで書いているため、<...> の箇所と #PJM 行は公式マニュアルで確認して書き換えること。
#
# 使い方:
#   pjsub jobs/wisteria_magpo.sh                                  # 短いテスト(既定)
#   pjsub -x MODE=full -x SEED=1 jobs/wisteria_magpo.sh           # 本番(論文設定)
#
# 課金グループ: gg45 / テスト用は share-debug、本番は share-short 等に変更 (pjstat --rsc で確認)
# #PJM 行の行末にコメントを書くと pjsub がエラーになるので、コメントは別行に書くこと
#PJM -g gg45
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=00:15:00
#PJM -N magpo
#PJM -j
#PJM -o logs/magpo_%j.out

set -euo pipefail

# --- 引数(pjsub -x で上書き可) ---
MODE="${MODE:-test}"                   # test | full
SEED="${SEED:-42}"
ENV="${ENV:-lbf}"
SCENARIO="${SCENARIO:-2s-8x8-2p-2f-coop}"
# 論文値(experiment_data/params.csv の magpo 行)に合わせる。タスクごとに変わる。
ALPHA="${ALPHA:-2}"
CLIP_GPO="${CLIP_GPO:-1.5}"            # params.csv の delta に対応する可能性が高いが未確認
ACTOR_LR="${ACTOR_LR:-0.0005}"

cd "${PJM_O_WORKDIR:-$PWD}"
mkdir -p logs

# --- 環境 ---
# 計算ノードは外部ネットワークに出られないことが多いので、
# 依存は事前にログインノードで `uv sync --extra cuda12` しておくこと。
# 必要ならここで module load (CUDA 等) を行う。例: module load cuda/12.x  (要確認)
PY="$PWD/.venv/bin/python"
if [ ! -x "${PY}" ]; then
  echo "ERROR: ${PY} not found. Run 'uv sync --extra cuda12' in $PWD on the login node." >&2
  exit 1
fi

# GPU メモリを JAX が全確保しないようにする(同一GPUで複数ジョブを走らせる場合に有効)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "== host: $(hostname)  mode: ${MODE}  seed: ${SEED}"
"${PY}" -c "import jax; print('jax devices:', jax.devices())"

if [ "${MODE}" = "test" ]; then
  # 速度測定用: ACTOR の Steps per second を見て本番の所要時間を見積もる
  NUM_UPDATES=20
  NUM_EVAL=2
else
  NUM_UPDATES=1220
  NUM_EVAL=122
fi

"${PY}" mava/systems/gpo/anakin/rec_magpo.py \
  env="${ENV}" env/scenario="${SCENARIO}" \
  arch.num_envs=64 \
  system.total_timesteps=null \
  system.num_updates="${NUM_UPDATES}" \
  arch.num_evaluation="${NUM_EVAL}" \
  system.seed="${SEED}" \
  system.alpha="${ALPHA}" \
  system.clip_gpo="${CLIP_GPO}" \
  system.actor_lr="${ACTOR_LR}" \
  logger.loggers.json.enabled=True
