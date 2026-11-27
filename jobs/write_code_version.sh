#!/bin/bash
# rsync で Wisteria に送る前にローカルで実行し、コードのバージョン(commit)を .code_version に書く。
# 送り先には .git が無いので、wisteria_coordsum.sh はこのファイルを読んで記録する。
#   使い方:  bash jobs/write_code_version.sh [コードのあるディレクトリ(既定: カレント)]
set -euo pipefail
cd "${1:-.}"
dirty=no
[ -n "$(git status --porcelain --untracked-files=no)" ] && dirty=yes
echo "commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=${dirty}" > .code_version
cat .code_version
