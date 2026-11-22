#!/bin/bash
# 論文 Figure (a) CoordSum の再現ジョブ。1ジョブ = 1手法 × (複数タスク × 複数seed) を同一GPUで並列実行。
# ハイパーパラメータは jobs/coordsum_overrides.py が experiment_data/params.csv から生成する。
#
# 使い方:
#   pjsub -x SYSTEM=magpo -x TASK=3x10-30 -x MODE=test jobs/wisteria_coordsum.sh   # 速度測定
#   pjsub -x SYSTEM=magpo jobs/wisteria_coordsum.sh                                # 本番(4タスク, seed 1)
#   bash jobs/submit_coordsum.sh                                                   # 全手法まとめて
#
# 注意: #PJM 行の行末にコメントを書くと pjsub がエラーになる。
#PJM -g gg45
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -N coordsum
#PJM -j
#PJM -o logs/coordsum_%j.out

set -euo pipefail

SYSTEM="${SYSTEM:-magpo}"        # magpo | mappo | happo | sable | mat
TASK="${TASK:-all}"              # all | 3x10-30 | 3x30-50 | 5x20-80 | 8x15-100
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-1}"
MAXPAR="${MAXPAR:-4}"            # 同時に走らせる run 数
MODE="${MODE:-full}"             # test(20更新) | full(論文設定)

if [ "${TASK}" = "all" ]; then TASKS="3x10-30 3x30-50 5x20-80 8x15-100"; else TASKS="${TASK}"; fi
SEEDS="$(seq "${SEED_START}" "${SEED_END}" | tr '\n' ' ')"

cd "${PJM_O_WORKDIR:-$PWD}"
mkdir -p logs
PY="$PWD/.venv/bin/python"
if [ ! -x "${PY}" ]; then
  echo "ERROR: ${PY} not found. Run 'uv sync --extra cuda12' in $PWD on the login node." >&2
  exit 1
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "== host: $(hostname) system: ${SYSTEM} tasks: ${TASKS} seeds: ${SEEDS} mode: ${MODE} maxpar: ${MAXPAR}"
"${PY}" -c "import jax; print('jax devices:', jax.devices())"

run_one() {
  local task="$1" seed="$2"
  local spec script
  spec="$("${PY}" jobs/coordsum_overrides.py "${SYSTEM}" "${task}")"
  script="$(echo "${spec}" | head -n 1)"
  local overrides=()
  while IFS= read -r line; do overrides+=("${line}"); done < <(echo "${spec}" | tail -n +2)
  if [ "${MODE}" = "test" ]; then
    overrides+=("system.num_updates=20" "arch.num_evaluation=2")
  fi
  "${PY}" "${script}" \
    env=coordsum env/scenario="${task}" \
    "${overrides[@]}" \
    system.total_timesteps=null \
    system.seed="${seed}" \
    logger.base_exp_path="results/coordsum_${SYSTEM}_${task}_s${seed}" \
    logger.loggers.json.enabled=True \
    > "logs/coordsum_${SYSTEM}_${task}_s${seed}_${PJM_JOBID:-local}.out" 2>&1
}

fail=0
running=0
for t in ${TASKS}; do
  for s in ${SEEDS}; do
    ( run_one "${t}" "${s}" ) &
    running=$((running + 1))
    if [ "${running}" -ge "${MAXPAR}" ]; then
      wait -n || fail=1
      running=$((running - 1))
    fi
  done
done
while [ "${running}" -gt 0 ]; do
  wait -n || fail=1
  running=$((running - 1))
done
echo "== done system=${SYSTEM} fail=${fail}"
exit "${fail}"
