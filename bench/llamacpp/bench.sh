#!/usr/bin/env bash
# llama.cpp (CUDA backend) on the same measurements as bench_engine:
#   prefill throughput for prompts of 128, 512 and 2048 tokens (llama-bench pp)
#   decode throughput for 128 tokens after 128, 512 and 2048 tokens of context (tg at depth)
# Everything on the GPU (-ngl 99), flash attention on, 5 repetitions.
#
#   bench/llamacpp/bench.sh GGUF LABEL [LLAMA_CPP_DIR]
#
# Writes bench/results/LABEL.jsonl (llama-bench's own JSON lines) and LABEL.txt.
# llama-bench reports mean and standard deviation over repetitions, where bench_engine
# reports median and range.
set -euo pipefail

gguf=${1:?gguf file}
label=${2:?label}
llama=${3:-$HOME/src/llama.cpp}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out="$repo/bench/results"
bin="$llama/build/bin/llama-bench"
mkdir -p "$out"
rm -f "$out/$label.jsonl"

q() { nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null | head -1; }
{
  echo "date          $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "llama.cpp     $(git -C "$llama" describe --tags --always) ($(git -C "$llama" rev-parse --short HEAD))"
  echo "model         $gguf"
  echo "gpu           $(q name), driver $(q driver_version)"
  echo "gpu before    $(q utilization.gpu) busy, $(q temperature.gpu) C, $(q power.draw)"
  echo
} > "$out/$label.txt"

common=(-m "$gguf" -ngl 99 -fa 1 -r 5)
"$bin" "${common[@]}" -p 128,512,2048 -n 0 -o jsonl >> "$out/$label.jsonl"
"$bin" "${common[@]}" -p 0 -n 128 -d 128,512,2048 -o jsonl >> "$out/$label.jsonl"

python3 - "$out/$label.jsonl" <<'EOF' | tee -a "$out/$label.txt"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"{'test':>16} | {'tok/s (mean +/- sd)':>22}")
for r in rows:
    test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']} @ d{r['n_depth']}"
    print(f"{test:>16} | {r['avg_ts']:>12.1f} +/- {r['stddev_ts']:.1f}")
EOF
