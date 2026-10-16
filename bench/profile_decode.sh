#!/usr/bin/env bash
# Profiles one decode step of the engine.
#
#   bench/profile_decode.sh MODEL_DIR BACKEND LABEL [CONTEXT [ENGINE FLAGS...]]
#   e.g. bench/profile_decode.sh models/llama-3.2-1b optimized int4 512 --weights int4
#
# Writes to bench/results/profiles/:
#   LABEL-ncu.csv     Nsight Compute metrics for every kernel launched in the step
#   LABEL-ncu.md      the same, summarised per kernel (bench/summarize_ncu.py)
#   LABEL-nsys.txt    Nsight Systems kernel summary for the step
# and keeps the full reports (LABEL.ncu-rep, LABEL.nsys-rep, gitignored) for the GUIs.
#
# Nsight Compute replays each kernel several times to collect all metrics and flushes the
# caches between replays, so its per-kernel times are cold-cache numbers; the Nsight Systems
# summary is from a normal run. Clocks are left alone (--clock-control none) so the numbers
# are comparable with bench_engine; Nsight Compute would otherwise lock them to base clocks.
set -euo pipefail

model=${1:?model dir}
backend=${2:?backend}
label=${3:?label}
ctx=${4:-512}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="$repo/bench/results/profiles"
bin="$repo/build/release/bench_engine"
mkdir -p "$out"
shift $(( $# < 4 ? $# : 4 ))
args=(--model "$model" --backend "$backend" --contexts "$ctx" --profile-step "$@")

# Both tools write SQLite or temp files that do not work reliably on a Windows drive mounted
# in WSL, so everything is produced in a local temp directory and copied over.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# --capture-range-end=stop lets the program exit normally; the default kills it (exit 143).
# --cuda-graph-trace=node reports the kernels inside a graph launch, not just the graph.
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --force-overwrite=true -o "$work/$label" "$bin" "${args[@]}" > "$work/nsys.log" 2>&1 || { cat "$work/nsys.log"; exit 1; }
nsys stats --force-export=true --report cuda_gpu_kern_sum --format table \
  "$work/$label.nsys-rep" > "$out/$label-nsys.txt" 2> "$work/stats.log" || { cat "$work/stats.log"; exit 1; }

# One full Nsight Compute pass (every section, for the GUI), then the raw metrics of that same
# report as CSV for bench/summarize_ncu.py.
if ! ncu --profile-from-start off --clock-control none --set full --force-overwrite \
  -o "$work/$label" "$bin" "${args[@]}" > "$work/ncu.log" 2>&1; then
  if grep -q ERR_NVGPUCTRPERM "$work/ncu.log"; then
    echo "Nsight Compute: no access to GPU performance counters (ERR_NVGPUCTRPERM)."
    echo "Enable them for all users in the NVIDIA Control Panel and re-run. Kept the nsys part:"
    cat "$out/$label-nsys.txt"
    exit 0
  fi
  cat "$work/ncu.log"
  exit 1
fi
ncu --import "$work/$label.ncu-rep" --page raw --csv > "$work/ncu.csv" 2> "$work/import.log" \
  || { cat "$work/import.log"; exit 1; }

cp "$work/ncu.csv" "$out/$label-ncu.csv"
cp "$work/$label.nsys-rep" "$work/$label.ncu-rep" "$out/"
python3 "$repo/bench/summarize_ncu.py" "$out/$label-ncu.csv" > "$out/$label-ncu.md"
cat "$out/$label-ncu.md"
