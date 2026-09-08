# `lib_sparse` architecture

## Design goal

The library targets activations whose zeros are known from a ReLU. A dense activation is expensive to retain for backward, especially when the FFN expansion dimension and batch/token count are large. `lib_sparse` stores only positive values plus enough metadata to reconstruct their positions and to apply the activation derivative.

The representation is intentionally simple and GPU-friendly. It avoids a general sparse-matrix format, per-value row/column indices, and CPU-side packing.

## Data representation

For a dense matrix `X[M, N]`, the default tile size is `BLOCK_M = 64` and `BLOCK_N = 64`. The tile grid is:

```text
grid_m = ceil(M / 64)
grid_n = ceil(N / 64)
tile_id = tile_m * grid_n + tile_n
```

Tiles are stored in grid-major order. Every tile has 4096 logical positions, including zero-padded positions at the right and bottom edges.

`BitsparseTensor` contains:

| Field | Meaning |
| --- | --- |
| `vals` | Positive values in tile-major order, either raw BF16/FP8 or a packed `uint8` stream. |
| `bitmask` | `uint8` mask with 1 bit per tile element; 512 bytes per 64×64 tile. Bit `f` is in byte `f // 8`, bit `f % 8`. |
| `prefix` | `uint32` array of length `num_tiles + 1`; `prefix[t]` is the logical value-stream offset of tile `t`. |
| `vals_offset` | Logical starting offset when `vals` is a shared `TensorBuffer`. |
| `scale` | Optional tensor-wise FP8 scale used by sparse arithmetic. |
| `shape`, `dtype` | Original logical matrix shape and reconstruction/storage dtype. |
| `grid_m`, `grid_n`, `BLOCK_M`, `BLOCK_N` | Geometry needed by the Triton kernels. |

The prefix array means a tile can find its values without searching. For tile `t`, the active value at local rank `r` is at `vals_offset + prefix[t] + r` in raw storage.

## Compression pipeline

The path in `dense_to_tilesparse` is:

```text
X[M,N]
  │
  ├─ _tile_pack_kernel ──> tile_counts[int32] + tile_bitmasks[uint8]
  │
  ├─ torch.cumsum(tile_counts)
  │                         └─> prefix[uint32]
  │
  └─ _compact_vals_kernel ─> compact positive value stream
                                │
                                └─ optional _pack_kernel
```

`_tile_pack_kernel` loads one complete tile, treats out-of-bounds positions as zero, computes `X > 0`, packs eight boolean values into each byte, and reduces the same boolean vector to a count.

The prefix is initialized as a device-side zero tensor. The cumsum writes into an `int32` view of `prefix[1:]`; this preserves the unsigned bit pattern while avoiding a scalar `prefix[0] = 0` assignment that would synchronize the host with CUDA.

`_compact_vals_kernel` loads one tile, expands its packed mask, computes a local rank with `tl.cumsum(mask) - 1`, and scatters active values into the global stream. Its base address is:

```text
prefix[tile_id] - prefix[first_tile] + vals_offset
```

That expression lets the same kernel write either a complete stream or a chunk-relative staging buffer.

## Bit-packed storage

When `pack_sbit=True`, the stream is packed continuously rather than storing one byte or word per value:

| Logical dtype | Stored bits/value | Reason |
| --- | ---: | --- |
| BF16 | 15 | Drop the sign bit from a non-negative 16-bit BF16 representation. |
| FP8 e4m3fn/e5m2 | 7 | Drop the sign bit from an 8-bit FP8 representation. |

The bitstream is read as `uint16` words. To decode a value at logical index `i`, the kernel computes:

```text
word_index = floor(i * bits / 16)
shift      = (i * bits) mod 16
value      = ((word[word_index] | word[word_index + 1] << 16) >> shift) & mask
```

Reading two words handles values crossing a word boundary. Standalone packed allocations are rounded to a four-byte boundary and include one guard word for these vectorized reads. Shared packed allocations align each tensor's logical start to eight values, which is byte-aligned for both 7-bit and 15-bit codecs.

Packed values are not numerically quantized by the packer. They are a bit-level representation of the already stored BF16 or FP8 value. FP8 quantization occurs in `to_fp8` or inside `MatmulFp8` before packing.

## Unpacking and arithmetic

`_unpack_batch_kernel` launches one program per tile in the selected row batch. It expands the bitmask, computes the same local rank, gathers values from the raw or packed stream, optionally squares them, and writes a dense `[batch_rows, K]` slice.

The public helpers build on that kernel:

- `unpack_batch_`: reconstruct values as stored.
- `unpack_relu2_batch_`: reconstruct `RELU2_SCALE * r²`; for FP8, it first promotes `r` to FP32/BF16-safe arithmetic and applies `scale²` outside the kernel.
- `AspB`: unpack sparse `B`, then compute `A @ B`.
- `spAB`: unpack sparse `A`, then compute `A @ B`.
- `AspRelu2B`: unpack `r`, form `RELU2_SCALE * r²`, then compute the product.

The optional `row_batch` in the sparse matmul helpers keeps only a tile-aligned row slice dense at a time. It reduces peak temporary VRAM but adds a Python loop and multiple matmul launches.

## ReLU FFN forward path

For `FFNRelu.apply(x, W1, W2)`:

```text
z = x @ W1.T
z = z + b1                    (optional)
r = ReLU(z)
save sparse(r) for backward
y = r @ W2.T
y = y + b2                    (optional)
```

For `FFNRelu2.apply`, the saved representation is still `r = ReLU(z)`, while the forward output uses:

```text
y = (RELU2_SCALE * r²) @ W2.T + b2
```

The forward matmul is dense. For FP8 mode, `MatmulFp8` converts BF16 operands as needed, applies tensor-wise scales, and returns BF16.

In the FP8 ReLU² path, the code saves an FP8 version of `r` for sparse backward, but squares the original dense `r` for the forward result before the forward matmul. This preserves the intended forward computation while accepting the stored FP8 representation in backward.

## Backward path

The custom autograd functions save the second-layer weight and a `BitsparseTensor` rather than a dense activation.

For ReLU:

```text
grad_W2 = AspB(grad_output.T, saved_r)
grad_z  = grad_output @ W2
grad_z  = grad_z * (saved_r > 0)
```

For ReLU², with `r = ReLU(z)`:

```text
grad_W2 = AspRelu2B(grad_output.T, saved_r)
grad_z  = (grad_output @ W2) * (2 * RELU2_SCALE * r)
```

The ReLU mask and ReLU² derivative are applied by in-place Triton kernels. The ReLU² kernel also reads the compact stored values rather than fully decompressing `r` for the derivative. If an FP8 scale exists, the wrapper applies the scale after the kernel.

The `backward` methods are marked `torch.compiler.disable`, so PyTorch's compiler does not attempt to trace these custom backward implementations. `to_fp8` is separately marked `torch.compile` and `no_grad`.

## Shared-buffer lifecycle

Each compression may either allocate its own value store or append to a `TensorBuffer`. In the shared case:

```text
TensorBuffer.vals  = one physical raw/packed allocation
TensorBuffer.offset = device scalar for the next logical value position
```

The returned sparse object keeps a cloned starting offset. The next compression updates the shared offset after its own prefix length. This supports a stack of saved activations without allocating a separate large values tensor for every layer.

The buffer must remain alive and unchanged until all consumers have finished. In training, resetting it between forward and backward would invalidate every saved sparse activation that points into it.

## Module relationships

```text
layers.py
  ├─ functions.dense_to_tilesparse
  ├─ sparse_matmul.AspB / AspRelu2B
  ├─ triton_operators.mask_with_bitmask_ / relu2_grad_sparse_
  └─ fp8.matmul / to_fp8

functions.py
  └─ triton_operators.tile_pack / compact_vals

triton_operators.py
  └─ triton_kernels.py + bitpacking.py
```

The package root does not re-export these names. The internal `src` modules are therefore part of the practical API even though they are implementation-oriented.
