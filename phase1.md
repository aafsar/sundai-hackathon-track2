# Phase 1: MMA Starter, Fast Path, And Scale Caching

Status: completed. GPU server result: correctness passed and Avg GEMM TOPs reached `108.77` on NVIDIA RTX A6000.

This phase is the first implementation slice of the CUDA INT4 GEMM optimization plan. It moves the solution from naive SIMT GEMM to an MMA tensor-core GEMM and adds a small set of low-risk overhead reductions plus scale caching.

## Hypothesis

The original `your_solution/kernel.cu` used a naive SIMT GEMM where each thread computed one output element by unpacking INT4 values in scalar code. The largest immediate improvement should come from using Ampere INT4 tensor cores through `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`.

After that jump, some overhead remains from:

- zero-filling the output tensor before writing every element,
- bounds predicates that are unnecessary for the fixed benchmark shapes,
- repeated global scale loads inside the per-warp N-tile loop,
- repeated `half` to `float` conversion for the same scales,
- scalar FP16 output stores.

Phase 1 groups the first set of fixes because they all keep the same high-level math and public API.

## Result Summary

Hardware and benchmark settings:

- GPU: NVIDIA RTX A6000
- Group size: `64`
- Score: Avg GEMM TOPs = `108.77`
- Previous target baseline from README: MMA starter around `58` Avg GEMM TOPs
- Improvement over MMA starter target: about `1.88x`

All correctness checks passed:

| Layer | Shape | Cosine | Threshold | Status |
|---|---:|---:|---:|---|
| `attn_to_qkv` | 4096x9216x3072 | 0.991494 | 0.989 | PASS |
| `attn_to_out` | 4096x3072x3072 | 0.992810 | 0.991 | PASS |
| `ff_up` | 4096x12288x3072 | 0.982393 | 0.978 | PASS |
| `ff_down` | 4096x3072x12288 | 0.982302 | 0.977 | PASS |

Performance:

| Layer | Quant GB/s | GEMM TOPs | Ref TOPs | Speedup vs naive ref |
|---|---:|---:|---:|---:|
| `attn_to_qkv` | 34.68 | 107.50 | 1.11 | 97.00x |
| `attn_to_out` | 35.07 | 107.39 | 1.11 | 96.72x |
| `ff_up` | 35.43 | 105.08 | 1.11 | 94.80x |
| `ff_down` | 12.27 | 115.13 | 1.08 | 106.54x |

Main conclusion:

- Phase 1 should be kept.
- The speedup confirms that tensor-core MMA plus cleanup and scale caching are beneficial.
- GEMM TOPs are now clustered between `105.08` and `115.13`, so the next phase should target a shared bottleneck rather than a single weak layer.
- Because `ff_down` has much larger `K` and still performs well, split-K is not the next priority.
- The next best target is packed-data memory layout, starting with offline weight packing because it is not timed.

## Files Changed

- `your_solution/kernel.cu`
- `your_solution/quantize.py`

No benchmark or reference files should be changed.

## Implemented Changes

### 1. Replaced naive GEMM with MMA tensor-core GEMM

The GEMM implementation now uses the same core shape as `reference/gemm_int4_mma.cu`:

- `BLOCK_M = 128`
- `BLOCK_N = 128`
- `BLOCK_K = 64`
- `NUM_WARPS = 8`
- `WARP_M = 16`
- `TILES_N = 8`
- `SMEM_STRIDE = 48`

The kernel uses:

- `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` on SM80+,
- fallback decomposed `m8n8k32` MMA path for older architecture compilation,
- `cp.async` for global-to-shared A/B tile copies,
- double-buffered shared memory for packed INT4 A/B tiles,
- direct shared-memory fragment loading into MMA register operands.

Why it should help:

- INT4 tensor cores perform many more integer dot products per cycle than scalar SIMT unpack/multiply loops.
- This is the expected jump from around naive baseline performance toward the MMA starter baseline.

Simple explanation:

- The naive version is like asking every CUDA thread to manually unpack tiny 4-bit numbers and multiply them one output cell at a time.
- The MMA version gives a whole small matrix multiply to tensor cores, which are special hardware units built exactly for dense matrix math.
- `m16n8k64` means one MMA instruction consumes a 16-row by 64-K A fragment and an 8-column by 64-K B fragment, then updates a 16x8 output fragment.
- Since our quantization group is also 64 elements wide, each MMA K step lines up naturally with one scale group.

Correctness note:

- MMA partials remain `int32`.
- Each 64-wide K tile corresponds to one quantization group when `group_size=64`.
- The `int32` group partial is scaled by A and B group scales before being accumulated into FP32 output accumulators.

Result interpretation:

- This is the main reason Phase 1 passed the old MMA starter range and reached `108.77` Avg GEMM TOPs.
- The result is much faster than the benchmark's naive reference, with around `95x` to `107x` speedup over the naive reference GEMM.

### 2. Changed GEMM output allocation to `torch::empty`

The GEMM wrapper now allocates the output tensor with `torch::empty` instead of zero-filling it.

Why it should help:

- The benchmark shapes are fully covered by the GEMM kernel.
- Zero-filling `C` before the kernel is unnecessary work and can add a timed GPU memset.

Simple explanation:

- `torch::zeros` allocates the output and fills it with zeros before the GEMM runs.
- But the GEMM writes every output element anyway.
- So the zero-fill is wasted memory bandwidth and wasted time.
- `torch::empty` allocates the output without clearing it, letting the kernel immediately write the real result.

Correctness risk:

- If a non-covered shape reaches the fast path, uninitialized output would be visible.
- The implementation guards the fast path with tile-alignment checks and keeps guarded stores in the generic path.

Result interpretation:

- This likely provided a small but reliable improvement.
- It is especially useful because output tensors are large: for example `ff_up` writes `4096 x 12288` FP16 values.

### 3. Added a benchmark-aligned fast path

The host wrapper computes:

- `fast_path = group_size == 64`
- `M % 128 == 0`
- `N % 128 == 0`
- `K % 64 == 0`

For the fixed benchmark shapes this should be true.

The fast path removes:

- A/B tile load predicates,
- epilogue bounds checks.

Why it should help:

- Less branch and predicate overhead in the hot GEMM path.
- Simpler generated code for the benchmark case.

Simple explanation:

- A generic kernel has to constantly ask, "am I outside the matrix boundary?"
- That is needed for arbitrary shapes, but the challenge shapes are clean multiples of the tile sizes.
- In the fast path, every 128x128 block is fully inside the output matrix, so those checks can be skipped.
- This means fewer instructions and fewer conditional paths while the GPU is doing the hot GEMM loop.

Correctness risk:

- The fast path assumes all CTA tiles are fully in-bounds.
- The four benchmark shapes satisfy this because `M=4096`, each `N` is divisible by 128, and each `K` is divisible by 64.

Result interpretation:

- The benchmark shapes all use the fast path.
- This helps explain why performance is consistent across the three `K=3072` layers despite different `N` sizes.

### 4. Preserved `int32` MMA accumulation discipline

The kernel still uses `int32` MMA fragments for each group:

- `int p0[4]`
- `int p1[4]`

It does not accumulate raw `int32` partials across multiple groups.

Why this is necessary:

- Different K groups have different A and B scales.
- Accumulating raw `int32` across groups and scaling once at the end would be mathematically wrong.

Simple explanation:

- The tensor core gives us an exact integer dot product for one 64-element group.
- But each 64-element group has its own scale, like its own unit conversion factor.
- We must convert each group's integer result into the correct FP32 value before adding it to the final answer.
- If we added many raw integer groups first and multiplied by one scale later, we would mix values with different units.

What changed around it:

- Scale products are applied as FP32 fused multiply-add contributions to final FP32 accumulators.
- This keeps correctness aligned with per-group quantization.

Result interpretation:

- The cosine results confirm this math is sound.
- All four layers passed, including the stricter `attn_to_out` threshold of `0.991`.

### 5. Changed activation scales to group-major physical layout

The online activation quantizer still returns a tensor shaped `[M, num_groups]`, but writes the raw contiguous buffer as:

```text
scales_A[group * M + row]
```

instead of:

```text
scales_A[row * num_groups + group]
```

Why it should help:

- For one CTA and one K group, GEMM needs scales for 128 adjacent M rows.
- Group-major layout makes those 128 scale loads contiguous.

Simple explanation:

- The public tensor still looks like `[M, num_groups]`, so the benchmark interface is unchanged.
- Internally, the raw memory is arranged as all rows for group 0, then all rows for group 1, and so on.
- During GEMM, a block works on one K group at a time and needs many row scales for that group.
- Group-major memory puts those needed scales next to each other, which is easier for the GPU to load efficiently.

Correctness risk:

- GEMM must read activation scales using the same raw group-major convention.

Result interpretation:

- Correctness passing confirms online activation quantization and GEMM agree on the new scale layout.
- Quantization throughput stayed around `35 GB/s` for the `K=3072` layers, so this layout change did not break the timed quantization path.

### 6. Changed weight scales to group-major physical layout

The offline weight quantizer still returns a tensor shaped `[N, num_groups]`, but its contiguous buffer is produced by:

```python
scale.squeeze(-1).half().transpose(0, 1).contiguous().view(N, num_groups)
```

GEMM reads it as:

```text
scales_B[group * N + col]
```

Why it should help:

- For one CTA and one K group, GEMM needs scales for 128 adjacent N columns.
- Group-major layout makes those 128 scale loads contiguous.
- Weight quantization is offline and untimed, so changing its physical layout is cheap.

Simple explanation:

- Weight scales have the same access pattern problem as activation scales.
- The GEMM block does not want "all groups for one column"; it wants "many columns for this one group."
- The offline quantizer rearranges scale memory into that order.
- Since offline quantization is not timed, we can spend extra Python-side work to make the timed CUDA GEMM simpler and faster.

Correctness risk:

- The public shape is intentionally misleading: shape remains `[N, num_groups]`, but raw memory is group-major.
- This is allowed only because the custom GEMM owns the interpretation.

Result interpretation:

- Correctness passing confirms offline weight scale layout and CUDA GEMM indexing match.
- This change is likely part of why the larger-N layers still stay near `105` to `108` TOPs.

### 7. Added shared-memory scale caching

For each K group, each CTA now loads:

- 128 A scales into `sScaleA`,
- 128 B scales into `sScaleB`.

These are converted from FP16 to FP32 once and reused across the CTA's MMA work.

Why it should help:

- The MMA starter loads B scales repeatedly inside the N-tile loop.
- Scale caching reduces global memory load count and repeated half-to-float conversions.
- Scale reads become coalesced due to group-major physical layout.

Simple explanation:

- Global memory is far away compared with shared memory.
- Without caching, the kernel repeatedly asks global memory for scale values that many lanes or N tiles reuse.
- With caching, one CTA loads the 128 row scales and 128 column scales for the current K group into shared memory.
- Threads then reuse those fast shared-memory copies while doing MMA work.
- The scales are also converted from FP16 to FP32 once, instead of converting the same values repeatedly.

Potential downside:

- Shared-memory footprint increases by `(128 + 128) * sizeof(float) = 1024` bytes.
- There is a synchronization cost per K tile.
- If scale loads were not a bottleneck, this could be neutral or slower.

Result interpretation:

- The final score suggests the tradeoff is positive overall.
- The shared-memory overhead did not prevent strong occupancy/performance on A6000 for these shapes.
- Since all layers landed in a tight TOPs band, scale handling is no longer obviously catastrophic, but profiling would be needed to know if more scale work remains.

### 8. Added `half2` vectorized output stores in the fast path

The fast path stores adjacent output columns as `half2`.

Why it should help:

- Each lane writes pairs of adjacent FP16 values.
- Fewer store instructions in the epilogue.

Simple explanation:

- The output is FP16, so two neighboring output values fit naturally into one `half2`.
- Instead of issuing two separate 16-bit stores, the fast path stores a pair together.
- This reduces epilogue instruction count and can make output writes cleaner.

Correctness risk:

- Requires adjacent output addresses and alignment.
- The fast-path column positions are adjacent pairs and benchmark `N` values are even and tile-aligned.

Result interpretation:

- This is probably a smaller gain than MMA or scale caching.
- It is still worth keeping because correctness passed and every benchmark layer uses the fast path.

## What Was Not Implemented In Phase 1

These are intentionally left for later phases:

- Tile-major packed B layout.
- Tile-major packed A layout.
- Shape-specific tile variants like `128x64x64`.
- CTA swizzle for L2 reuse.
- 3-stage `cp.async` pipeline.
- Shared-memory stride experiments.
- Split-K for `ff_down`.
- Persistent CTA or stream-K scheduling.

## Reproduction Commands

Run a quick compile/correctness/performance smoke test:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cuda-challenge
python benchmark.py --warmup 2 --iters 5
```

Then run the normal benchmark:

```bash
./benchmark.sh
```

If the data is missing, `benchmark.sh` should download it. If running `benchmark.py` directly and data is missing, run:

```bash
python download_data.py
```

## Raw Result

```text
GPU: NVIDIA RTX A6000
Group size: 64
Cosine thresholds: {'attn_to_qkv': 0.989, 'attn_to_out': 0.991, 'ff_up': 0.978, 'ff_down': 0.977}

Running offline weight quantization...
  attn_to_qkv: [9216, 3072] -> packed [9216, 1536]
  attn_to_out: [3072, 3072] -> packed [3072, 1536]
  ff_up: [12288, 3072] -> packed [12288, 1536]
  ff_down: [3072, 12288] -> packed [3072, 6144]

CORRECTNESS CHECK:
  attn_to_qkv     (4096x9216x3072):  cosine=0.991494  threshold=0.989  [PASS]
  attn_to_out     (4096x3072x3072):  cosine=0.992810  threshold=0.991  [PASS]
  ff_up           (4096x12288x3072):  cosine=0.982393  threshold=0.978  [PASS]  (ref cuda: 0.9824 WARNING)
  ff_down         (4096x3072x12288):  cosine=0.982302  threshold=0.977  [PASS]  (ref cuda: 0.9823 WARNING)

PERFORMANCE BENCHMARK:
  Layer               M      N      K   Quant GB/s   GEMM TOPs    Ref TOPs   Speedup
  attn_to_qkv      4096   9216   3072      34.68      107.50        1.11     97.00x
  attn_to_out      4096   3072   3072      35.07      107.39        1.11     96.72x
  ff_up            4096  12288   3072      35.43      105.08        1.11     94.80x
  ff_down          4096   3072  12288      12.27      115.13        1.08    106.54x

SCORE: Avg GEMM TOPs = 108.77
```

## Final Result Table

| Layer | Correct? | Cosine | Quant GB/s | GEMM TOPs | Ref TOPs | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| `attn_to_qkv` | yes | 0.991494 | 34.68 | 107.50 | 1.11 | 97.00x |
| `attn_to_out` | yes | 0.992810 | 35.07 | 107.39 | 1.11 | 96.72x |
| `ff_up` | yes | 0.982393 | 35.43 | 105.08 | 1.11 | 94.80x |
| `ff_down` | yes | 0.982302 | 12.27 | 115.13 | 1.08 | 106.54x |

Average GEMM TOPs: `108.77`.

Decision: keep Phase 1 and build Phase 2 on top of it.

## What The Results Mean

The result is strong because every layer passed correctness and all GEMM TOPs are close together. A single layer is not obviously broken.

The jump from the MMA starter target of `58` to `108.77` suggests Phase 1 did more than simply copy the MMA starter. The most likely contributors are:

- no output memset from `torch::empty`,
- faster fully-aligned benchmark path,
- better scale memory access through group-major scale layout,
- less repeated scale loading through shared-memory scale caching,
- simpler epilogue stores through `half2`.

The `ff_down` layer has `K=12288`, four times larger than the other layers, and it achieved the highest GEMM TOPs at `115.13`. That means longer K loops are not currently the obvious weakness. Split-K should stay deferred.

The `ff_up` layer is the lowest at `105.08`, but the gap is small. Since `ff_up` has the largest `N`, this points toward global memory movement and B-side layout as a plausible next bottleneck. That matches the planned next phase: tile-major packed weight layout.

## Next Step

Move to Phase 2: packed weight tile-major layout.

Reason:

- Weight quantization is offline and untimed.
- Repacking B is less risky than repacking online activations.
- The current GEMM still loads B tiles from row-major `[N, K/2]` storage, where adjacent loader threads can touch memory separated by `K/2` bytes.
- Tile-major B layout should make GEMM's B tile loads more contiguous and reduce global memory transaction waste.
