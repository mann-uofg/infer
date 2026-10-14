"""Summarises Nsight Compute CSV output (--csv --page raw) for one decode step.

Kernels are grouped by name (the same kernel runs once per layer) and reported with total
time, share of the step, DRAM traffic, achieved bandwidth and occupancy.

    python3 bench/summarize_ncu.py bench/results/profiles/naive-ncu.csv
"""

import csv
import sys
from collections import defaultdict


def short_name(name: str) -> str:
    # "void infer::kernels::(anonymous namespace)::MatmulNaiveKernel(const float *, ...)"
    # -> "MatmulNaiveKernel"; template arguments are kept.
    name = name.split("(")[0] if "<" not in name.split("(")[0] else name.split(">(")[0] + ">"
    return name.split("::")[-1].replace("void ", "")


def main(path: str) -> None:
    with open(path, newline="") as f:
        lines = [line for line in f if line.startswith('"')]
    rows = list(csv.DictReader(lines))
    # The second row holds units.
    units = rows[0]
    rows = [r for r in rows[1:] if r.get("Kernel Name")]

    def num(r, key):
        v = r[key].replace(",", "")
        return float(v) if v not in ("", "n/a") else 0.0

    time_unit = units["gpu__time_duration.sum"]
    scale_ns = {"ns": 1.0, "usecond": 1e3, "us": 1e3, "msecond": 1e6, "ms": 1e6, "nsecond": 1.0}[
        time_unit
    ]
    byte_scale = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}

    groups = defaultdict(lambda: {"n": 0, "ns": 0.0, "read": 0.0, "write": 0.0, "dram_pct": [],
                                  "sm_pct": [], "occ": [], "launch": ""})
    for r in rows:
        g = groups[short_name(r["Kernel Name"])]
        g["n"] += 1
        g["ns"] += num(r, "gpu__time_duration.sum") * scale_ns
        g["read"] += num(r, "dram__bytes_read.sum") * byte_scale[units["dram__bytes_read.sum"]]
        g["write"] += num(r, "dram__bytes_write.sum") * byte_scale[units["dram__bytes_write.sum"]]
        g["dram_pct"].append(num(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed"))
        g["sm_pct"].append(num(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed"))
        g["occ"].append(num(r, "sm__warps_active.avg.pct_of_peak_sustained_active"))
        g["launch"] = (f'{int(num(r, "launch__grid_size"))}x{int(num(r, "launch__block_size"))}'
                       f', {int(num(r, "launch__registers_per_thread"))} regs')

    total_ns = sum(g["ns"] for g in groups.values())
    total_bytes = sum(g["read"] + g["write"] for g in groups.values())
    print(f"One decode step: {len(rows)} kernel launches, {total_ns / 1e3:.1f} us of kernel time "
          f"(serialized, cold cache), {total_bytes / 1e6:.1f} MB DRAM traffic, "
          f"{total_bytes / total_ns:.1f} GB/s overall.\n")
    print("| kernel | launches | time us | share | DRAM MB | GB/s | DRAM % peak (mean) "
          "| SM % peak (mean) | occupancy % (mean) | last launch |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["ns"]):
        mb = (g["read"] + g["write"]) / 1e6
        mean = lambda xs: sum(xs) / len(xs)
        print(f'| {name} | {g["n"]} | {g["ns"] / 1e3:.1f} | {100 * g["ns"] / total_ns:.1f}% '
              f'| {mb:.1f} | {(g["read"] + g["write"]) / g["ns"]:.1f} | {mean(g["dram_pct"]):.1f} '
              f'| {mean(g["sm_pct"]):.1f} | {mean(g["occ"]):.1f} | {g["launch"]} |')


if __name__ == "__main__":
    main(sys.argv[1])
