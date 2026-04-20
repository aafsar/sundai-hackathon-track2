# Phase 2: Packed Weight Tile-Major Layout

Status: implemented locally; waiting for GPU server benchmark results.

This phase changes the physical layout of offline packed weights so the GEMM
B tile loader reads each 128-column by 64-K tile from a compact contiguous
region. Public tensor shapes and wrapper signatures stay unchanged.

## Hypothesis

Phase 1 still stores packed weights in row-major `[N, K/2]` order. During a
CTA B tile load, the kernel needs 128 adjacent output columns for one 64-wide
K group. With row-major storage, adjacent B rows are separated by `K/2` bytes,
so the cooperative loader gathers from many distant locations.

Offline weight quantization is not timed, so it can prepack weights into the
order the timed GEMM consumes.

## Baseline

Phase 1 result on NVIDIA RTX A6000:

- Correctness: all four layers passed.
- Avg GEMM TOPs: `108.77`.

Per-layer Phase 1 GEMM TOPs:

| Layer | GEMM TOPs |
|---|---:|
| `attn_to_qkv` | 107.50 |
| `attn_to_out` | 107.39 |
| `ff_up` | 105.08 |
| `ff_down` | 115.13 |

## Implemented Layout

The returned `weight_packed` tensor still has public shape `[N, K/2]` and dtype
`uint8`.

For `group_size == 64`, `N % 128 == 0`, and `K % 64 == 0`, its raw contiguous
buffer is stored as:

```text
[k_group][n_tile][n_in_tile][packed_byte_in_64_group]
```

where:

- `k_group = K / 64`
- `n_tile = N / 128`
- `n_in_tile = 0..127`
- `packed_byte_in_64_group = 0..31`

The flat raw byte offset is:

```text
((k_group * N + n_tile * 128 + n_in_tile) * 32) + packed_byte
```

This preserves the standard INT4 byte convention:

```text
low nibble = even K element
high nibble = odd K element
signed range [-8, 7]
```

Weight scales remain in Phase 1 group-major physical layout:

```text
scales_B[group * N + col]
```

## Code Changes

- `your_solution/quantize.py`
  - Quantizes weights the same way as Phase 1.
  - Reorders only packed weight bytes for the guarded tile-major case.
  - Keeps row-major packed output for fallback shapes.
  - Keeps group-major weight scales unchanged.
- `your_solution/kernel.cu`
  - Adds a `B_TILE_MAJOR` kernel template.
  - Uses tile-major B source addresses when the host wrapper detects the
    guarded layout case.
  - Keeps the old row-major B source address for fallback.
  - Leaves shared-memory B layout, `load_b_frag`, MMA math, scale caching, CTA
    tile sizes, output allocation, and epilogue stores unchanged.

## Expected Effect

- More contiguous global B tile reads.
- Better global memory coalescing for B loads.
- Largest expected benefit on large-N layers:
  - `attn_to_qkv`
  - `ff_up`

## Correctness Risks

- A mismatch between Python offline B packing and CUDA B loader indexing causes
  immediate correctness failure.
- The public `[N, K/2]` shape no longer describes row-major physical memory for
  guarded benchmark shapes.
- Non-guarded shapes must continue to use row-major packed B.

## Validation Commands

Quick smoke test on the GPU server:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cuda-challenge
python benchmark.py --warmup 2 --iters 5
```

Full benchmark:

```bash
./benchmark.sh
```

Required correctness:

```text
attn_to_qkv cosine > 0.989
attn_to_out cosine > 0.991
ff_up       cosine > 0.978
ff_down     cosine > 0.977
```

Compare the final Avg GEMM TOPs against Phase 1's `108.77`.
