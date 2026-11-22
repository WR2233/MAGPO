#!/bin/bash
# Wisteria/BDEC-01 (Aquarius, A100) 用 PJM ジョブスクリプトの雛形。
# 記憶ベースで書いているため、<...> の箇所と #PJM 行は公式マニュアルで確認して書き換えること。
#
# 使い方:
#   pjsub jobs/wisteria_magpo.sh                                  # 短いテスト(既定)
#   pjsub -x MODE=full -x SEED=1 jobs/wisteria_magpo.sh           # 本番(論文設定)
#
#PJM -g <GROUP>                  # 課金グループ名 (例: gxxxxx)
#PJM -L rscgrp=<RSCGRP>          # 例: share-short(テスト) / share(本番)など。要確認
#PJM -L gpu=1                    # GPU 1枚。share系グループ向けの指定と記憶している。要確認
#PJM -L elapse=00:30:00          # walltime。本番は実測の steps/s から決めて延ばす
#PJM -N magpo
#PJM -j                          # 標準エラーを標準出力に統合
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
source .venv/bin/activate

# GPU メモリを JAX が全確保しないようにする(同一GPUで複数ジョブを走らせる場合に有効)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "== host: $(hostname)  mode: ${MODE}  seed: ${SEED}"
python -c "import jax; print('jax devices:', jax.devices())"

if [ "${MODE}" = "test" ]; then
  # 速度測定用: ACTOR の Steps per second を見て本番の所要時間を見積もる
  NUM_UPDATES=20
  NUM_EVAL=2
else
  NUM_UPDATES=1220
  NUM_EVAL=122
fi

python mava/systems/gpo/anakin/rec_magpo.py \
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
