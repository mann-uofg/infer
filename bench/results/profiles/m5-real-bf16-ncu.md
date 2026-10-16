One decode step: 117 kernel launches, 4209.9 us of kernel time (serialized, cold cache), 2549.5 MB DRAM traffic, 605.6 GB/s overall.

| kernel | launches | time us | share | DRAM MB | GB/s | DRAM % peak (mean) | SM % peak (mean) | occupancy % (mean) | last launch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gemv bf16, swiglu, 2 rows/warp | 16 | 1849.7 | 43.9% | 1110.9 | 600.6 | 92.9 | 7.7 | 73.3 | 1024x256, 44 regs |
| Gemv bf16, += residual, 1 rows/warp | 32 | 1185.3 | 28.2% | 680.6 | 574.2 | 82.6 | 6.0 | 53.2 | 256x256, 40 regs |
| Gemv bf16, store, 2 rows/warp | 17 | 952.2 | 22.6% | 735.8 | 772.7 | 80.2 | 7.7 | 42.0 | 8016x256, 44 regs |
| AttentionDecodeCombineKernel | 16 | 91.9 | 2.2% | 2.8 | 30.0 | 4.1 | 0.9 | 4.0 | 32x64, 40 regs |
| AttentionDecodeSplitKernel<64, 4> | 16 | 79.3 | 1.9% | 18.3 | 231.0 | 29.1 | 21.7 | 14.9 | 128x128, 51 regs |
| RopeKvDecodeKernel | 16 | 40.4 | 1.0% | 0.5 | 13.1 | 1.8 | 0.2 | 16.0 | 7x256, 24 regs |
| EmbeddingKernel | 1 | 4.4 | 0.1% | 0.0 | 1.7 | 0.2 | 0.0 | 15.6 | 1x256, 18 regs |
| ArgmaxPartialKernel | 1 | 3.2 | 0.1% | 0.6 | 186.0 | 21.4 | 5.4 | 23.6 | 126x256, 20 regs |
| ArgmaxFinishKernel | 1 | 1.9 | 0.0% | 0.1 | 35.6 | 4.1 | 0.0 | 2.2 | 1x1, 16 regs |
| AdvancePositionKernel | 1 | 1.5 | 0.0% | 0.0 | 7.3 | 0.9 | 0.0 | 2.2 | 1x1, 16 regs |
