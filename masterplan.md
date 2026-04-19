# CUDA INT4 GEMM Master Plan

This document is the self-contained handoff for optimizing this repository's CUDA Challenge solution. A new conversation should be able to read this file plus the current `your_solution/` files and continue the work without needing prior chat context.

## Challenge Context

The challenge is W4A4 quantized GEMM for real FLUX.1-schnell model layers.

The benchmark has two parts:

1. Offline weight quantization in `your_solution/quantize.py`.
   - Not timed.
   - Converts FP16 weights `[N, K]` into packed signed INT4 plus FP16 scales.
2. Online forward path in `your_solution/kernel.cu`.
   - Timed.
   - Quantizes FP16 activations `[M, K]` into packed signed INT4 plus FP16 scales.
   - Runs INT4 GEMM: `C[M, N] = A[M, K] @ B[N, K]^T`.

Scoring is the average GEMM TOPs across four fixed shapes:

| Layer | M | N | K |
|---|---:|---:|---:|
| `attn_to_qkv` | 4096 | 9216 | 3072 |
| `attn_to_out` | 4096 | 3072 | 3072 |
| `ff_up` | 4096 | 12288 | 3072 |
| `ff_down` | 4096 | 3072 | 12288 |

Correctness must pass per-layer cosine similarity thresholds against FP16 matmul:

| Layer | Threshold |
|---|---:|
| `attn_to_qkv` | 0.989 |
| `attn_to_out` | 0.991 |
| `ff_up` | 0.978 |
| `ff_down` | 0.977 |

Important constraints:

- Only edit `your_solution/quantize.py` and `your_solution/kernel.cu` for solution code.
- Do not modify `reference/`, `benchmark.py`, or `benchmark.sh`.
- Keep these public C++ wrapper signatures unchanged:
  - `std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size);`
  - `torch::Tensor gemm_int4_custom(torch::Tensor A_packed, torch::Tensor B_packed, torch::Tensor scales_A, torch::Tensor scales_B, int group_size);`
- Public tensor shapes should remain benchmark-compatible:
  - A packed: `[M, K/2]` `uint8`
  - B packed: `[N, K/2]` `uint8`
  - A scales: `[M, K/group_size]` `float16`
  - B scales: `[N, K/group_size]` `float16`
- Internal physical layout inside those tensors may change if `quantize.py`, online quantization, and GEMM all agree.
- Default target is `group_size=64`, which matches `BLOCK_K=64` and one scale group per MMA K tile.

## Baseline To Understand

The useful starting point is `reference/gemm_int4_mma.cu`, not the naive SIMT GEMM.

The MMA starter uses:

- CTA tile: `BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=64`.
- 8 warps per CTA.
- Ampere INT4 tensor-core instruction:
  - `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`
- `int32` MMA fragment accumulation for each 64-wide K group.
- FP32 accumulation across quantization groups after applying A/B scales.
- `cp.async` global-to-shared copies.
- Double-buffered shared memory for packed A/B tiles.

Why this matters:

- Naive SIMT assigns ordinary CUDA threads to output elements and leaves tensor cores idle.
- MMA uses hardware tensor cores and is the first large speedup step.
- After MMA, the likely bottlenecks move to data movement, scale handling, register pressure, shared-memory behavior, and epilogue stores.

Known benchmark baseline from README:

| Baseline | Approx Avg GEMM TOPs on RTX A6000 |
|---|---:|
| MMA starter | 58 |
| Optimized reference target | 305 |

Benchmarking is done on the remote GPU server.

## Collaboration Workflow

We optimize in phases. Each phase is a small, measurable set of changes with one main hypothesis.

For every phase:

1. Write a phase document, named `phaseN.md`.
   - Example: `phase1.md`, `phase2.md`, `phase3.md`.
   - Include the hypothesis, exact code changes, expected performance effect, correctness risks, and server commands.
2. Implement that phase's code changes.
   - Keep changes scoped.
   - Avoid mixing unrelated experiments in one phase unless they are intentionally grouped.
3. User runs the phase on the GPU server.
   - First quick run:
     ```bash
     source $HOME/miniconda3/etc/profile.d/conda.sh
     conda activate cuda-challenge
     python benchmark.py --warmup 2 --iters 5
     ```
   - Then full run:
     ```bash
     ./benchmark.sh
     ```
4. User shares the full benchmark output in the next conversation.
5. Analyze the results and update that same `phaseN.md`.
   - Add compile status.
   - Add correctness table.
   - Add per-layer TOPs.
   - Add average TOPs.
   - Explain which changes likely helped or hurt.
   - Decide whether to keep, modify, split, or revert the phase.
6. Only after the phase is understood, start the next phase.

Server access note:

- Use the push/pull workflow: local code changes are pushed, the user pulls on the GPU server, runs benchmarks, and shares results.

Recommended result format to paste back:

```text
GPU:
Commit / branch:
Command:

CORRECTNESS CHECK:
  attn_to_qkv ...
  attn_to_out ...
  ff_up ...
  ff_down ...

PERFORMANCE BENCHMARK:
  Layer            M      N      K   Quant GB/s   GEMM TOPs   Ref TOPs  Speedup
  ...

SCORE: Avg GEMM TOPs = ...
Any compile/runtime errors:
```

## Phase Roadmap

### Phase 1: MMA Starter Plus Low-Risk Cleanup And Scale Caching

Status: completed; details and analysis are in `phase1.md`.

Goals:

- Replace naive GEMM with MMA tensor-core GEMM.
- Remove unnecessary output zero-fill.
- Add fixed-shape fast path for benchmark-aligned shapes.
- Cache scales in shared memory per CTA/K group.
- Store scale tensors in group-major physical order while preserving public shapes.
- Use `half2` vectorized stores in the fast-path epilogue.

Expected effect:

- Large jump from naive SIMT toward MMA starter performance.
- Lower overhead from no output memset.
- Fewer GEMM-side global scale loads and half-to-float conversions.
- Less branch/predicate overhead in the benchmark fast path.
- Slightly better output store efficiency.

Primary risk:

- Scale layout must be consistent between offline weight quantization, online activation quantization, and GEMM.
- Shared-memory scale caching adds synchronization and shared-memory footprint; it may help or hurt depending on whether scale loads were a real bottleneck.

Measured result:

- GPU: NVIDIA RTX A6000
- Correctness: all four layers passed
- Avg GEMM TOPs: `108.77`
- Approx improvement over README MMA starter target of `58`: `1.88x`

Decision:

- Keep Phase 1.
- Do not split or revert the phase.
- Use Phase 1 as the new baseline for future work.

### Phase 2: Packed Weight Tile-Major Layout

Hypothesis:

The current GEMM still reads packed weights from row-major `[N, K/2]` storage. During one CTA's B tile load, the kernel wants 128 nearby output columns for one 64-wide K group. In row-major storage, those columns are separated by `K/2` bytes, so adjacent loader threads do not read one compact contiguous region.

Since weight quantization is offline and not timed, Phase 2 should prepack B into the memory order the GEMM wants.

Plan:

- Create `phase2.md` before implementing code.
- Keep the public `weight_packed` tensor shape `[N, K/2]`.
- Change only the physical memory order returned by `your_solution/quantize.py`.
- Store raw `weight_packed` memory grouped by:
  - K group,
  - N tile,
  - column inside the 128-column tile,
  - packed bytes inside the 64-wide K group.
- Update the GEMM B tile loader in `your_solution/kernel.cu` to read this tile-major physical layout.
- Keep Phase 1's scale layout, scale caching, fast path, and MMA math unchanged.
- Add a guarded fallback or explicit benchmark-shape assumption if needed.

Expected effect:

- More contiguous B tile reads.
- Better global memory coalescing.
- Fewer wasted memory transactions.
- Higher GEMM TOPs, especially for large-N layers like `attn_to_qkv` and `ff_up`.

Risks:

- Any mismatch between offline B layout and the GEMM B loader causes correctness failure.
- The public tensor shape will no longer describe the raw memory order.
- If B memory movement is not the current bottleneck, the gain may be small.

Validation:

- Run the normal benchmark and compare against Phase 1's `108.77` Avg GEMM TOPs.
- Require all four cosine thresholds to pass.
- Pay special attention to `attn_to_qkv` and `ff_up`, because those have the largest `N` values.

### Phase 3: Packed Activation Tile-Major Layout

Hypothesis:

After B layout is improved, A tile loads may become the next memory bottleneck. Online quantization currently writes packed activations in row-major `[M, K/2]` order, while GEMM wants a CTA-local 128-row by 64-K group tile.

Plan:

- Keep returned A packed shape `[M, K/2]`.
- Make online quantization write raw memory grouped by:
  - K group,
  - M tile,
  - row inside the 128-row tile,
  - packed bytes inside the 64-wide K group.
- Update GEMM A tile loader to read from this physical layout.
- Track both `Quant GB/s` and `GEMM TOPs`, because activation quantization is timed separately but the score is Avg GEMM TOPs.

Expected effect:

- More contiguous A tile loads in GEMM.
- Higher GEMM TOPs if A global-load efficiency is limiting.

Risks:

- Quantization may become slower.
- Correctness risk is high if A physical indexing is wrong.

### Phase 4: Tile Size And Shape-Specific Dispatch

Hypothesis:

The current `128x128x64` CTA tile has strong data reuse but high register pressure. Some shapes, especially `N=3072`, may prefer smaller N tiles or different M/N balance.

Variants to test:

- `128x128x64`: current baseline.
- `128x64x64`: fewer accumulators, likely better occupancy.
- `64x128x64`: lower per-CTA work, possible occupancy improvement.
- `128x256x64`: more reuse, likely high register pressure; test only if register count permits.

Plan:

- Implement one variant per phase or per clearly separated compile-time branch.
- Dispatch by shape only after data shows a variant wins for a layer family.
- Keep `BLOCK_K=64` for default `group_size=64`.

Expected effect:

- Better balance between tensor-core occupancy, shared-memory use, and register pressure.

### Phase 5: Pipeline, Shared-Memory, And CTA Scheduling

Hypothesis:

After scale and coalescing changes, remaining stalls may be from memory latency, shared-memory bank conflicts, or poor L2 reuse between CTAs.

Experiments:

- Compare 2-stage vs 3-stage `cp.async` buffering.
- Test shared-memory strides: `32`, `48`, `64` bytes.
- Add CTA swizzles:
  - identity order,
  - grouped N tiles for A reuse,
  - grouped M tiles for B reuse,
  - shape-specific swizzle for `N=3072` vs large-N layers.

Expected effect:

- Higher tensor-core active percentage.
- Better L2 hit rate.
- Fewer memory stalls.

### Phase 6: Advanced Scheduling Only If Profiling Justifies It

Only consider these if earlier phases plateau and profiling shows a clear need:

- Split-K for `ff_down` with `K=12288`.
- Persistent CTA scheduling.
- Stream-K scheduling.
- CUTLASS exploration.

Risks:

- Split-K needs extra workspace and a reduction kernel.
- Persistent/stream-K scheduling is complex and easy to make slower.
- cuBLAS/CUTLASS plain INT4 GEMM does not directly solve per-group A/B scale accumulation.

## Profiling Guidance

Use normal benchmark output first. If Nsight Compute is available, collect:

- Tensor-core utilization / pipe tensor active.
- DRAM throughput and global load efficiency.
- L2 hit rate.
- Shared-memory bank conflicts.
- Achieved occupancy.
- Registers per thread.
- Stall reasons.

Interpretation:

- If tensor-core utilization is low and memory stalls are high, prioritize layout and pipeline work.
- If occupancy is low and registers are high, prioritize tile-size variants.
- If scale loads/conversions are visible, prioritize scale caching/layout.
- If only one layer is weak, consider shape-specific dispatch.

## Invariants To Preserve

- INT4 packing convention:
  - low nibble = even element,
  - high nibble = odd element,
  - signed INT4 range `[-8, 7]`.
- Symmetric per-group quantization:
  - `scale = max(abs(x)) / 7`,
  - round-to-nearest,
  - clamp to `[-8, 7]`.
- For `group_size=64`, each MMA K tile maps to exactly one quantization group.
- Do not accumulate raw `int32` dot products across multiple scale groups unless the scale math is changed correctly. Different groups have different A/B scales, so each group must be scaled before contributing to the final FP32 sum.
- Every optimization must pass all four cosine thresholds before performance matters.
