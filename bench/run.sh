#!/usr/bin/env bash
# Runs a benchmark binary and saves its output with the conditions it ran under.
#
#   bench/run.sh LABEL bench_engine --model models/random-1b --backend naive ...
#   bench/run.sh LABEL bench_kernels --peak-gbps 851.5
#
# Writes bench/results/LABEL.txt (environment header + output) and, for binaries that take
# --json, bench/results/LABEL.jsonl.
set -euo pipefail

label=${1:?label}
binary=${2:?binary}
shift 2

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="$repo/bench/results"
mkdir -p "$out"
txt="$out/$label.txt"
jsonl="$out/$label.jsonl"
rm -f "$jsonl"

q() { nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null | head -1; }
{
  echo "date          $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "commit        $(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet || echo '+dirty')"
  echo "gpu           $(q name), driver $(q driver_version)"
  echo "toolkit       $(nvcc --version | grep -o 'release [0-9.]*')"
  echo "host          WSL2 $(uname -r)"
  echo "clocks before sm $(q clocks.sm) / mem $(q clocks.mem) (max sm $(q clocks.max.sm), not locked)"
  echo "gpu before    $(q temperature.gpu) C, $(q power.draw), $(q memory.used) in use"
  echo "command       $binary $*"
  echo
} > "$txt"

extra=()
case "$binary" in
  bench_engine|bench_kernels) extra=(--json "$jsonl") ;;
esac
"$repo/build/release/$binary" "$@" "${extra[@]}" | tee -a "$txt"
echo >> "$txt"
echo "clocks after  sm $(q clocks.sm) / mem $(q clocks.mem)" >> "$txt"
