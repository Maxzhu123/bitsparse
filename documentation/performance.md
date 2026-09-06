# Performance and optimization notes

`lib_sparse` is optimized around activation-storage bandwidth, GPU launch geometry, and backward memory pressure. The following are the important performance techniques and the costs they introduce.

## The central trade-off

The representation can reduce the stored activation value payload, but the current high-level path does not perform a sparse GEMM. It still has to:

1. compute the dense forward matmul;
2. build the mask, prefix, and compact stream;
3. unpack a dense temporary for sparse weight-gradient matmuls; and
4. run the dense input-gradient matmul.

Sparse execution can therefore be slower for small tensors or low sparsity. Its primary win is lower peak VRAM and the ability to keep more large batches/tokens in memory. The benchmark should measure forward, backward, compression, decompression, and peak memory separately.

## Tile geometry

The fixed 64×64 tile has 4096 elements. That choice gives the Triton kernels a regular program shape and makes a tile's metadata compact:

- 512 bytes for the bitmask;
- one count during compression;
- one prefix entry for its stream start;
- a single `tl.cumsum` over a 4096-element mask for rank calculation.

The same geometry is used by tile packing, compaction, unpacking, and the gradient kernels. Edge tiles are handled with masked loads/stores and zero-filled out-of-bounds positions; no special remainder kernel is needed.

## Bitmask plus prefix instead of per-value coordinates

The format stores one bit per logical position and one prefix entry per tile, not a row and column index for every active value. A tile can recover the compact-stream rank of an element with a local prefix count. This reduces metadata and makes the gather/scatter operations regular.

It also means that the format is a good fit for dense-ish ReLU activations with structured GPU tiles, but it is not a general sparse matrix format. Every tile still incurs a 512-byte mask and a prefix entry, even when it contains very few values.

## 7/15-bit continuous packing

ReLU guarantees non-negative stored values, so the sign bit is known to be zero and is not written. The remaining bits are concatenated across byte and word boundaries:

```text
BF16 raw: 16 bits/value -> packed: 15 bits/value
FP8 raw :  8 bits/value -> packed:  7 bits/value
```

This avoids wasting the unused sign bit and does not require an index table for the packed stream. The decoder reads adjacent `uint16` words, so a value crossing a word boundary still needs only two vectorized loads.

The `pack_sbit` mode has several implementation details that preserve throughput:

- standalone allocations are four-byte aligned and include a guard word;
- shared streams align each tensor start to eight logical values, so each independent pack launch starts on a byte boundary;
- packing is done on GPU by `_pack_kernel`, not by copying values to the CPU;
- the packed path uses a reusable raw staging buffer rather than allocating a separate raw stream for every chunk.

The cost is an extra packing/unpacking pass and more complicated address arithmetic. For BF16, packing alone only changes the value payload from 2 bytes to roughly 1.875 bytes, so the mask/prefix overhead and launch cost matter. FP8 changes the value payload from 1 byte to roughly 0.875 bytes.

## Chunked packed writes

The packed path processes at most `_PACK_CHUNK_TILES = 2**16` tiles per launch group. Each chunk is first compacted to workspace offset zero and then packed into its final stream position.

Chunking keeps individual launches and staging buffers bounded and allows a single workspace to be reused. Standalone allocations use the prefix array to calculate exact chunk lengths after determining the total nonzero count. Shared-buffer allocations avoid a host-side capacity calculation by using a dense upper bound for the workspace, which keeps the shared-buffer path asynchronous but can reserve more temporary workspace than the actual nonzero count.

## Avoiding accidental CUDA synchronizations

Several choices specifically protect the asynchronous GPU pipeline:

- The prefix array is created with `torch.zeros` and cumsum writes into its `int32` view. This avoids `prefix[0] = 0`, whose scalar assignment can synchronize.
- Shared-buffer offsets live in device tensors. `clone()` captures the starting offset on-device and `offset.copy_(...)` advances it without a Python integer round trip.
- Standalone compression does need `int(tile_prefix[-1].item())` to allocate an exact output. That is an intentional synchronization/accuracy trade-off; the shared-buffer path can use an upper bound instead.
- `nnz()`, `sparsity_ratio()`, `repr()`, and `vram_size()` are diagnostic conveniences, not hot-path methods. They read or convert device scalars and can synchronize.

## Kernel fusion and in-place work

The kernels combine related operations to avoid intermediate tensors:

- `_tile_pack_kernel` computes the positive mask, packs its bytes, and reduces its count in one tile program.
- `_compact_vals_kernel` expands the mask, computes ranks, and scatters values in one tile program.
- `_unpack_batch_kernel` decodes the mask/value stream, optionally squares values, and stores the dense tile.
- ReLU masking and ReLU² derivative kernels load and overwrite the gradient in place.

The Python layer also uses in-place `relu_`, `square_`, `mul_`, `addmm_`, and gradient updates where the autograd design permits it. This reduces peak allocations, but callers should not assume the inputs are preserved: the layer mutates intermediate tensors such as the preactivation after it has saved what backward needs.

The ReLU² gradient kernel is non-idempotent because it multiplies the gradient by `2 * RELU2_SCALE * r`. Its Triton autotune declaration uses `restore_value=["grad_ptr"]` so benchmark candidates do not compound their transformations across timing iterations.

## Triton autotuning

The hot kernels have small candidate sets covering 2, 4, 8, and sometimes 16 warps, plus one to three pipeline stages. Autotune keys are chosen according to the work shape:

| Kernel family | Main key inputs |
| --- | --- |
| Tile pack and compaction | `M`, `N` (packing also keys on codec/number of tiles) |
| Unpack | sparse tile width, output `K`, batch rows, packed/FP8/square flags, codec |
| ReLU mask | `M`, `N` |
| ReLU² gradient | `M`, `N`, packed/FP8 flags, codec |

The comments in `triton_kernels.py` show the intended tuning logic: unpack is memory-bound, compaction is dominated by the 4096-element scan/scatter, and elementwise operations use one tile per program. New tile sizes or GPU architectures should be benchmarked rather than assumed to use the same winning configuration.

## FP8-specific optimizations

`MatmulFp8` uses `torch.nn.functional.scaled_mm` with tensor-wise scales. It saves either:

- the original BF16/FP8 operands and reconverts them during backward (`CACHE_FP8_MATMUl=False`, the default), or
- the FP8 operands and scales directly (`CACHE_FP8_MATMUl=True`).

The second mode uses more VRAM but avoids repeated conversion during backward. The setting is global and is intentionally a memory/speed switch.

The FP8 wrapper also avoids unnecessary layout copies:

- `A` is made row-major contiguous only if its stride requires it;
- on Ada-class GPUs, `B` is converted to the column-major layout expected by cuBLASLt;
- on Blackwell-class GPUs, that reformat is skipped because row-major `B` is accepted.

The architecture check is evaluated when `fp8.py` is imported and looks only at CUDA device 0. Mixed-device applications need to account for that limitation.

## Shared storage and activation lifetime

One `TensorBuffer` can hold active values from several FFN layers. This reduces allocator traffic and makes the total value storage a single allocation. The trade-off is manual capacity management and a strict lifetime rule: the buffer cannot be reset or reused until every saved sparse activation has finished its backward use.

The packed allocator aligns each appended tensor to eight logical values. This wastes at most seven logical positions per tensor, but it prevents two independently launched pack operations from sharing a byte and avoids read/modify/write hazards at stream boundaries.

## Row-batched unpacking

`AspB`, `spAB`, and `AspRelu2B` accept `row_batch`. With the default `0`, the complete sparse operand is unpacked once. With a positive value, the code:

```text
select whole 64-row tile groups
unpack one group
multiply the matching dense slice
accumulate/store the result
```

This lowers peak temporary memory. It does not make the computation sparse and usually increases launch and loop overhead. The high-level FFN wrappers do not expose `row_batch`, so callers needing this trade-off must use the lower-level helpers or extend the layer interface.

## Memory accounting

For `nnz` active values and `T = grid_m * grid_n` tiles, the logical footprint reported by `BitsparseTensor.vram_size()` is approximately:

```text
raw:    nnz * sizeof(storage_dtype) + T * (BLOCK_M * BLOCK_N / 8) + (T + 1) * 4
packed: ceil(nnz * bits_per_value / 8)  + T * (BLOCK_M * BLOCK_N / 8) + (T + 1) * 4
```

The actual standalone packed allocation is rounded to four bytes and includes an additional four-byte guard word. A shared buffer's physical allocation also contains values belonging to other sparse tensors and any alignment gaps, so `vram_size()` should be treated as a per-object logical estimate rather than a complete shared-buffer accounting.

## What to benchmark

For useful comparisons, report at least:

- peak forward and backward VRAM;
- compression time;
- unpack time;
- total forward/backward time;
- raw versus packed value bytes;
- actual positive-value ratio, not just the requested data sparsity;
- BF16 versus FP8 numerical error.

Warm up Triton autotuning and synchronize CUDA before reading wall-clock timings. The repository's validation and benchmark scripts show the intended CUDA-event and correctness-check pattern, but the benchmark helper currently passes an outdated `storage_dtype` argument to `dense_to_tilesparse` and should be updated to the current signature first.

