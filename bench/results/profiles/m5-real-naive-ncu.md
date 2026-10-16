One decode step: 166 kernel launches, 54669.8 us of kernel time (serialized, cold cache), 2949.6 MB DRAM traffic, 54.0 GB/s overall.

| kernel | launches | time us | share | DRAM MB | GB/s | DRAM % peak (mean) | SM % peak (mean) | occupancy % (mean) | last launch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AttentionNaiveKernel<64> | 16 | 25466.6 | 46.6% | 116.3 | 4.6 | 0.5 | 0.1 | 2.1 | 1x64, 167 regs |
| MatmulNaiveKernel | 65 | 24688.0 | 45.2% | 2810.8 | 113.9 | 13.7 | 3.3 | 17.5 | 501x256, 39 regs |
| RmsNormNaiveKernel | 33 | 4359.1 | 8.0% | 19.7 | 4.5 | 0.5 | 0.1 | 2.1 | 1x256, 39 regs |
| ArgmaxKernel | 1 | 46.5 | 0.1% | 0.5 | 11.8 | 1.3 | 0.1 | 66.4 | 1x1024, 16 regs |
| RopeNaiveKernel | 16 | 37.8 | 0.1% | 0.5 | 14.1 | 1.7 | 0.4 | 16.4 | 5x256, 26 regs |
| KvCacheWriteKernel | 16 | 35.4 | 0.1% | 0.3 | 8.9 | 1.1 | 0.1 | 16.2 | 2x256, 22 regs |
| SwiGluKernel | 16 | 29.1 | 0.1% | 1.3 | 46.3 | 5.4 | 1.5 | 15.8 | 32x256, 20 regs |
| EmbeddingKernel | 1 | 4.4 | 0.0% | 0.0 | 2.7 | 0.3 | 0.0 | 15.6 | 1x256, 18 regs |
| StoreTokenKernel | 1 | 1.6 | 0.0% | 0.0 | 3.7 | 0.4 | 0.0 | 2.2 | 1x1, 16 regs |
| SetPositionKernel | 1 | 1.2 | 0.0% | 0.0 | 4.0 | 0.5 | 0.0 | 2.2 | 1x1, 16 regs |
