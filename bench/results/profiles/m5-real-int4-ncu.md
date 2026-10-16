One decode step: 117 kernel launches, 2235.6 us of kernel time (serialized, cold cache), 732.4 MB DRAM traffic, 327.6 GB/s overall.

| kernel | launches | time us | share | DRAM MB | GB/s | DRAM % peak (mean) | SM % peak (mean) | occupancy % (mean) | last launch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gemv int4, swiglu, 4 rows/warp | 16 | 761.8 | 34.1% | 308.4 | 404.9 | 81.5 | 18.3 | 31.3 | 512x256, 86 regs |
| Gemv int4, += residual, 4 rows/warp | 32 | 624.9 | 28.0% | 193.6 | 309.9 | 58.7 | 10.4 | 16.5 | 64x256, 86 regs |
| Gemv int4, store, 4 rows/warp | 17 | 584.7 | 26.2% | 207.8 | 355.4 | 55.5 | 11.3 | 22.7 | 4008x256, 86 regs |
| AttentionDecodeCombineKernel | 16 | 103.5 | 4.6% | 3.0 | 28.9 | 6.1 | 0.8 | 4.3 | 32x64, 40 regs |
| AttentionDecodeSplitKernel<64, 4> | 16 | 101.8 | 4.6% | 18.3 | 179.5 | 37.5 | 16.9 | 15.1 | 128x128, 51 regs |
| RopeKvDecodeKernel | 16 | 46.8 | 2.1% | 0.6 | 11.8 | 2.5 | 0.2 | 15.3 | 7x256, 24 regs |
| EmbeddingKernel | 1 | 4.8 | 0.2% | 0.0 | 1.5 | 0.2 | 0.0 | 16.8 | 1x256, 18 regs |
| ArgmaxPartialKernel | 1 | 3.4 | 0.2% | 0.7 | 205.9 | 23.9 | 5.1 | 34.8 | 126x256, 20 regs |
| AdvancePositionKernel | 1 | 2.0 | 0.1% | 0.0 | 1.1 | 0.3 | 0.0 | 2.1 | 1x1, 16 regs |
| ArgmaxFinishKernel | 1 | 2.0 | 0.1% | 0.0 | 1.3 | 0.2 | 0.0 | 2.1 | 1x1, 16 regs |
