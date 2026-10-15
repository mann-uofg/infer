#!/usr/bin/env bash
# Every benchmark behind the numbers in the README, in one go. Run on an idle GPU (close
# games and anything else that renders); the script refuses to start if the GPU is busy.
#
#   bench/run_all.sh MODEL_DIR GGUF_DIR PREFIX
#
#   MODEL_DIR  Hugging Face checkpoint (models/random-1b or the real one)
#   GGUF_DIR   output of bench/llamacpp/convert.sh for the same checkpoint
#   PREFIX     results go to bench/results/PREFIX-*.{txt,jsonl}
set -euo pipefail

model=${1:?model dir}
gguf=${2:?gguf dir}
prefix=${3:?prefix}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
py="${PYTHON:-$HOME/venvs/infer/bin/python}"

busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$busy" -gt 10 ]; then
  echo "GPU is ${busy}% busy; close whatever is using it and try again." >&2
  exit 1
fi

ctx=(--contexts 128,512,2048 --generate 128 --runs 5)

# Peak bandwidth first: every percentage below is relative to it.
bench/run.sh "$prefix-membw" membw | tail -1
peak=$(grep -o 'read [0-9.]*' "bench/results/$prefix-membw.txt" | awk '{print $2}' | sort -n | tail -1)
echo "peak read bandwidth: $peak GB/s"

bench/run.sh "$prefix-kernels" bench_kernels --peak-gbps "$peak" > /dev/null

run_engine() {
  local label=$1
  shift
  bench/run.sh "$prefix-engine-$label" bench_engine --model "$model" "${ctx[@]}" \
    --peak-gbps "$peak" --label "$label" "$@" | tail -3
}
run_engine naive --backend naive
run_engine bf16-nograph --graph off
run_engine bf16
run_engine int8 --weights int8
run_engine int4 --weights int4
# Groups of 32 like Q4_0, and an int8 head for a Q6_K one: the same perplexity as llama.cpp's
# Q4_0 file on Llama-3.2-1B, and about the same bytes per token.
run_engine int4-int8head --weights int4 --lm-head int8
run_engine int4-g128 --weights int4 --int4-group 128

for q in bf16 Q8_0 Q4_0; do
  bench/llamacpp/bench.sh "$gguf/model-$q.gguf" "$prefix-llamacpp-$q" | tail -6
done

for attn in sdpa eager; do
  rm -f "bench/results/$prefix-hf-$attn.jsonl"
  "$py" bench/hf_baseline.py --model "$model" --attn "$attn" --runs 5 \
    --json "bench/results/$prefix-hf-$attn.jsonl" | tee "bench/results/$prefix-hf-$attn.txt"
done
