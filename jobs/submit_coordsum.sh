#!/bin/bash
# CoordSum 再現ジョブを手法ごとに投入する(ログインノードで実行)。
#   bash jobs/submit_coordsum.sh                       # magpo mappo happo sable mat、seed 1
#   SYSTEMS="magpo mat" bash jobs/submit_coordsum.sh   # 手法を絞る
#   MODE=test bash jobs/submit_coordsum.sh             # 速度測定(各20更新)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for sys in ${SYSTEMS:-magpo mappo happo sable mat}; do
  pjsub -x SYSTEM="${sys}" -x MODE="${MODE:-full}" -x SEED_START="${SEED_START:-1}" -x SEED_END="${SEED_END:-1}" jobs/wisteria_coordsum.sh
done
