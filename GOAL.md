# Hack Goal

Improve the average GEMM TOPs score for the CUDA Challenge W4A4 quantized GEMM benchmark while understanding the kernel changes well enough to explain why they improve performance.

## Objective

Build a faster implementation of the timed online path:

1. Quantize FP16 activations to packed signed INT4 on the GPU.
2. Run INT4 GEMM for the four fixed FLUX.1-schnell layer shapes.
3. Preserve correctness by meeting each layer's cosine similarity threshold.

The score is the average GEMM TOPs across:

| Layer | M | N | K |
|---|---:|---:|---:|
| attn_to_qkv | 4096 | 9216 | 3072 |
| attn_to_out | 4096 | 3072 | 3072 |
| ff_up | 4096 | 12288 | 3072 |
| ff_down | 4096 | 3072 | 12288 |

Only `your_solution/quantize.py` and `your_solution/kernel.cu` should be edited.

## What To Understand First

- The data format: two signed INT4 values packed per `uint8`, with per-group FP16 scales.
- The math: `C[M,N] = A[M,K] @ B[N,K]^T`, where both A and B are quantized INT4 and dequantized by scale during accumulation/output.
- The baseline gap: naive SIMT is around 1.1 TOPs, the MMA starter is around 58 TOPs, and the optimized reference target is around 305 TOPs on an RTX A6000.
- The accuracy constraint: optimizations are useful only if all four cosine thresholds still pass.

## Optimization Approach

Start from the MMA reference path, not the naive one. The naive kernel assigns ordinary CUDA threads to output elements and leaves tensor cores idle; the MMA kernel uses INT4 tensor-core instructions, which is the first large jump in TOPs.

Then iterate on the real bottlenecks:

1. **Tiling**
   - Choose M/N/K tiles that maximize tensor-core work per block.
   - Reuse loaded A and B tiles across many output elements.
   - Better tiling improves arithmetic intensity, so more time is spent on MMA instructions instead of global memory traffic.

2. **Shared memory staging**
   - Stage packed INT4 tiles and scales in shared memory.
   - Use double buffering or `cp.async` where appropriate to overlap global memory loads with compute.
   - This improves TOPs by reducing stalls while tensor cores wait for data.

3. **Scale handling**
   - Broadcast or cache per-group scales efficiently at warp/block scope.
   - Avoid repeatedly loading the same scales for every output element.
   - This matters because INT4 math is fast enough that scale loads and conversion overhead can become visible.

4. **Register-level accumulation and output layout**
   - Keep accumulators in registers and avoid unnecessary transposes or stores.
   - Arrange each warp's output fragment so stores are coalesced.
   - This improves effective bandwidth and reduces instruction overhead after MMA.

5. **Activation quantization**
   - Keep the online A quantization fast and format-compatible with the GEMM kernel.
   - Use parallel reductions for per-group max-abs scale computation, then pack two clamped INT4 values per byte.
   - Quantization is timed, so GEMM improvements should not be offset by slow activation preprocessing.

6. **Offline weight quantization**
   - Use `your_solution/quantize.py` to produce a layout that the CUDA kernel can consume efficiently.
   - Since this step is not timed, it can prepack or reorder weights to reduce runtime indexing and improve memory coalescing.
   - Any layout change must still match the fixed wrapper contract and correctness checks.

## Iteration Plan

1. Run `./benchmark.sh` to establish the current baseline.
2. Replace the naive GEMM with the MMA starter if not already done.
3. Benchmark each change across all four shapes, not just one layer.
4. Track both average GEMM TOPs and per-layer cosine similarity.
5. Keep a short note for every attempted optimization:
   - what changed,
   - which bottleneck it targets,
   - TOPs before/after,
   - whether correctness passed.

## Success Criteria

The hack is successful if:

- The solution beats the MMA starter baseline by a meaningful margin.
- Every layer passes the benchmark's cosine similarity threshold.
- The final implementation can be explained in terms of data movement, tensor-core utilization, scale handling, and output storage.
- Each chosen optimization has a measured reason for staying in the code.
