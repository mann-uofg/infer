"""Roofline plot from bench_kernels results.

    python bench/plot_roofline.py bench/results/m5-kernels.jsonl docs/roofline.png \\
        --peak-gbps 851.5 --peak-tflops 55.4

x: arithmetic intensity, flops per byte of DRAM traffic (minimum bytes the kernel must move).
y: achieved GFLOP/s. The roof is min(peak compute, intensity x peak bandwidth). Bandwidth is
the measured peak (bench/membw); compute is the fp32 CUDA-core peak, 8960 lanes x 2 flops x
the max boost clock nvidia-smi reports (3.09 GHz), so it is an upper bound, not a measurement.

Decode GEMVs sit far left (about 1 flop per byte for bf16): no amount of arithmetic tuning
moves them; only bytes do, which is why quantization speeds decode up. Prefill GEMMs sit far
right, under the compute roof.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Flops of each benchmarked op, keyed by the name bench_kernels prints.
H, I, V, Q, KV = 2048, 8192, 128256, 2048, 512
GEMV_FLOPS = {
    "gemv qkv 3072x2048": 2 * (Q + 2 * KV) * H,
    "gemv o 2048x2048": 2 * H * Q,
    "gemv gate_up 16384x2048": 2 * 2 * I * H,
    "gemv down 2048x8192": 2 * H * I,
    "gemv lm_head 128256x2048": 2 * V * H,
}
# (N, K) of each prefill projection.
GEMM_SHAPES = {"qkv": (Q + 2 * KV, H), "o": (H, Q), "gate_up": (2 * I, H), "down": (H, I)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--peak-gbps", type=float, required=True)
    ap.add_argument("--peak-tflops", type=float, required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.results)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [0.1, 1e4]
    ridge = args.peak_tflops * 1e3 / args.peak_gbps
    ax.plot([xs[0], ridge, xs[1]], [xs[0] * args.peak_gbps, args.peak_tflops * 1e3,
                                    args.peak_tflops * 1e3], color="black", linewidth=1.5)
    ax.text(0.15, 0.15 * args.peak_gbps * 1.3, f"{args.peak_gbps:.0f} GB/s (measured)",
            rotation=35, fontsize=9)
    ax.text(ridge * 1.5, args.peak_tflops * 1e3 * 1.08,
            f"{args.peak_tflops:.1f} TFLOPS fp32 (theoretical)", fontsize=9)

    styles = {"naive": ("o", "#bbbbbb"), "optimized": ("o", "#1f77b4"),
              "int8": ("s", "#2ca02c"), "int4": ("^", "#d62728"),
              "tiled": ("D", "#9467bd"), "cublas-f32": ("x", "#8c564b"),
              "cublas-bf16": ("+", "#e377c2")}
    seen = set()
    for r in rows:
        name, variant = r["kernel"], r["variant"]
        if variant not in styles:
            continue
        if name in GEMV_FLOPS:
            flops = GEMV_FLOPS[name]
            intensity = flops / r["bytes"]
            gflops = flops / (r["us"] * 1e-6) / 1e9
        elif "flops" in r:  # prefill GEMM "gemm <proj> T=<t>"
            proj = name.split()[1]
            t = int(name.split("T=")[1])
            n, k = GEMM_SHAPES[proj]
            a_elem = 2 if variant == "cublas-bf16" else 4  # cuBLAS bf16 reads bf16 activations
            bytes_ = n * k * 2 + t * k * a_elem + t * n * 4  # W once, A in, C out
            intensity = r["flops"] / bytes_
            gflops = r["tflops"] * 1e3
        else:
            continue
        marker, color = styles[variant]
        label = variant if variant not in seen else None
        seen.add(variant)
        ax.scatter(intensity, gflops, marker=marker, color=color, label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xs)
    ax.set_ylim(10, args.peak_tflops * 1e3 * 3)
    ax.set_xlabel("arithmetic intensity (flops / byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("RTX 5070 Ti: decode GEMVs vs prefill GEMMs")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
