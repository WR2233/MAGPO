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
RESULTS_ROOT="${RESULTS_ROOT:-results}"   # 結果の保存先(実験条件ごとに分けるため)

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

# 実行したコードのバージョン(commit)を記録する。git が無い場所(rsync で送った作業ディレクトリ)では
# 送信前にローカルで作った .code_version(jobs/write_code_version.sh)を使う。
code_version() {
  if git rev-parse --git-dir >/dev/null 2>&1; then
    local dirty="no"
    [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && dirty="yes"
    echo "commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=${dirty}"
  elif [ -f .code_version ]; then
    cat .code_version
  else
    echo "commit=unknown"
  fi
}
CODE_VERSION="$(code_version)"

echo "== host: $(hostname) system: ${SYSTEM} tasks: ${TASKS} seeds: ${SEEDS} mode: ${MODE} maxpar: ${MAXPAR}"
echo "== code: ${CODE_VERSION}"
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
  # 手元(Mac)での動作確認用: 環境数を絞る
  if [ -n "${NUM_ENVS:-}" ]; then
    overrides+=("arch.num_envs=${NUM_ENVS}")
  fi
  # run ごとの専用 TMPDIR(同時JITコンパイル時の /tmp 競合による ptxas エラー対策)
  local tmpd="$PWD/tmp/${PJM_JOBID:-local}_${SYSTEM}_${task}_s${seed}"
  mkdir -p "${tmpd}"
  export TMPDIR="${tmpd}"
  local log="logs/coordsum_${SYSTEM}_${task}_s${seed}_${PJM_JOBID:-local}.out"
  : > "${log}"
  echo "== code: ${CODE_VERSION} script: ${script} system: ${SYSTEM} job: ${PJM_JOBID:-local}" >> "${log}"
  # 結果ディレクトリにも、どのコードで実行したかを残す
  local rdir="${RESULTS_ROOT}/coordsum_${SYSTEM}_${task}_s${seed}"
  mkdir -p "${rdir}"
  {
    echo "${CODE_VERSION}"
    echo "script=${script}"
    echo "system=${SYSTEM} task=${task} seed=${seed} job=${PJM_JOBID:-local} date=$(date +%Y-%m-%dT%H:%M:%S)"
    echo "overrides=${overrides[*]}"
  } > "${rdir}/code_version.txt"
  local attempt
  for attempt in 1 2 3; do
    echo "== attempt ${attempt}" >> "${log}"
    if "${PY}" "${script}" \
      env=coordsum env/scenario="${task}" \
      "${overrides[@]}" \
      system.total_timesteps=null \
      system.seed="${seed}" \
      logger.base_exp_path="${RESULTS_ROOT}/coordsum_${SYSTEM}_${task}_s${seed}" \
      logger.loggers.json.enabled=True \
      >> "${log}" 2>&1; then
      rm -rf "${tmpd}"
      return 0
    fi
    echo "== attempt ${attempt} failed, retrying" >> "${log}"
    sleep 10
  done
  return 1
}

# 失敗した run は FAILDIR にファイルを作って記録する(wait -n を使わないので bash 3.2 でも動く)
FAILDIR="$(mktemp -d "${PWD}/logs/.fail_XXXXXX")"
for t in ${TASKS}; do
  for s in ${SEEDS}; do
    while [ "$(jobs -r | wc -l)" -ge "${MAXPAR}" ]; do sleep 5; done
    ( run_one "${t}" "${s}" || touch "${FAILDIR}/${t}_s${s}" ) &
    sleep "${STAGGER:-20}"
  done
done
wait
fail="$(ls "${FAILDIR}" | wc -l | tr -d ' ')"
if [ "${fail}" -gt 0 ]; then echo "== FAILED runs: $(ls "${FAILDIR}" | tr '\n' ' ')"; fi
rm -rf "${FAILDIR}"
echo "== done system=${SYSTEM} failed_runs=${fail}"
[ "${fail}" -eq 0 ]
