"""
Per-tile compressed sparse format — shared kernel library.

A dense 2D tensor X ∈ R^{M×N} is partitioned into a grid of tiles, each
of shape [BLOCK_M × BLOCK_N].  Every tile is independently compressed:

  bitmask  — uint8 packed bitmask (8 bits per byte), TILE_BYTES bytes/tile.
             Row-major within the tile, so bit at flat offset f lives in
             byte f//8 at bit position f%8.  A set bit (1) means the
             element is nonzero after ReLU: X[i,j] > 0.

  vals     — a uint8 stream containing the positive nonzero values with their
             zero sign bits omitted (15 bits/value), in grid-major order.

  prefix   — int32 prefix sum of per-tile nonzero counts: prefix[t] is
             the starting offset of tile t's values inside vals.
             prefix[num_tiles] equals the total number of nonzero values.
"""

import triton
import triton.language as tl

from src.code.bitpacking import load_15bit_at_indices


_MATMUL_CONFIGS = [
    triton.Config({"BLOCK_K": 32}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_K": 64}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=2),
]

# ═══════════════════════════════════════════════════════════════════════════════
# _tile_pack_kernel
#   Computes:  bitmask[t] = pack(X_tile > 0)    ∀ tile t
#              counts[t]  = ||X_tile > 0||₀     (number of positive entries)
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _tile_pack_kernel(
    dense_ptr,          # pointer to dense input X ∈ R^{M×N}
    tile_counts_ptr,    # output: int32[n_tiles] nonzero counts per tile
    tile_bitmasks_ptr,  # output: uint8[n_tiles × TILE_BYTES] packed bitmasks
    M, N,               # dimensions of X
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,    # = BLOCK_M × BLOCK_N
    TILE_BYTES: tl.constexpr,    # = TILE_NUMEL // 8
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.num_programs(1) + pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    tile = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_flat = tl.reshape(tile, (TILE_NUMEL,))
    nz = (tile_flat > 0.0)                      # boolean: ReLU mask for this tile

    # Pack 8 bools → 1 uint8: reshape to [TILE_BYTES, 8], shift each bit
    # position by j ∈ {0..7}, sum across bit axis.
    #   bytes_val[b] = Σ_{j=0}^{7} nz[8b + j] · 2^j
    nz_reshaped = tl.reshape(nz, (TILE_BYTES, 8))
    bit_weights = tl.arange(0, 8)[None, :]
    bytes_val = tl.sum(nz_reshaped.to(tl.int32) << bit_weights, 1).to(tl.uint8)
    tl.store(tile_bitmasks_ptr + pid * TILE_BYTES + tl.arange(0, TILE_BYTES), bytes_val)

    nnz = tl.sum(nz.to(tl.int32))
    tl.store(tile_counts_ptr + pid, nnz)


# ═══════════════════════════════════════════════════════════════════════════════
# _compact_vals_kernel
#   Given prefix[t] = Σ_{i=0}^{t-1} count[i] (exclusive prefix sum),
#   scatters tile t's positive values into a contiguous 16-bit staging tensor.
#   A following bandwidth-oriented kernel packs that staging tensor to 15 bits.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _compact_vals_kernel(
    dense_ptr,          # input:  dense X ∈ R^{M×N}
    tile_prefix_ptr,    # input:  int32[n_tiles+1] exclusive prefix sum of counts
    vals_out_ptr,       # output: compact bf16/fp16 values
    layer_offset_ptr,   # input:  int32[1] global offset where this layer starts
    M, N, grid_n,       # dimensions and tile grid
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.load(layer_offset_ptr)
    base = tl.load(tile_prefix_ptr + pid) + offset

    tile_m = pid // grid_n
    tile_n = pid % grid_n

    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v_2d = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    v = tl.reshape(v_2d, (TILE_NUMEL,))

    nz = (v > 0.0).to(tl.int32)

    # rank[i] is the logical value index within this tile.
    ranks = tl.cumsum(nz, 0) - 1
    tl.store(vals_out_ptr + base + ranks, v, mask=(nz == 1))


@triton.jit
def _compact_vals_15bit_kernel(
    dense_ptr,
    tile_prefix_ptr,
    vals_out_ptr,
    layer_offset_ptr,
    M, N, grid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    """Compact directly into a zeroed packed stream for shared buffers.

    Shared buffers cannot allocate a value-count-sized staging tensor, so this
    allocation-free variant uses atomic word writes.
    """
    pid = tl.program_id(0)
    offset = tl.load(layer_offset_ptr)
    base = tl.load(tile_prefix_ptr + pid) + offset

    tile_m = pid // grid_n
    tile_n = pid % grid_n
    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v_2d = tl.load(
        dense_ptr + offs,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
        other=0.0,
    )
    v = tl.reshape(v_2d, (TILE_NUMEL,))
    nz = (v > 0.0).to(tl.int32)
    ranks = tl.cumsum(nz, 0) - 1
    value_bits = v.to(tl.uint16, bitcast=True).to(tl.int32) & 0x7FFF
    value_indices = base + ranks
    within_group = value_indices % 32
    word_indices = ((value_indices // 32) * 15
                    + (within_group * 15) // 32)
    shifts = (within_group * 15) % 32

    low_words = value_bits << shifts
    tl.atomic_or(vals_out_ptr + word_indices, low_words, mask=(nz == 1))

    crosses_word = shifts > 17
    high_shifts = tl.where(crosses_word, 32 - shifts, 1)
    high_words = value_bits >> high_shifts
    tl.atomic_or(
        vals_out_ptr + word_indices + 1,
        high_words,
        mask=(nz == 1) & crosses_word,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# _unpack_batch_kernel
#   Reconstructs dense tiles from the sparse representation.
#   For each tile t in a batch of rows:
#     D_tile = 0
#     for each nonzero position i in tile t (from bitmask[t]):
#         D_tile[i] = vals[prefix[t] + rank[i]]
#   This computes: D_rowslice = gather(vals, bitmask, prefix)
#   where D_rowslice ∈ R^{batch_rows × K} is written into dense_ptr.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _unpack_batch_kernel(
    vals_ptr,           # input:  compact nonzero values (bf16)
    bitmask_ptr,        # input:  uint8 packed bitmasks
    prefix_ptr,         # input:  int32[n_tiles+1] exclusive prefix sum
    layer_offset_ptr,   # input:  int32[1] global offset for this layer
    dense_ptr,          # output: dense bf16 buffer of shape [batch_rows, K]
    first_m_tile,       # first row-tile in this batch
    grid_n_sparse, K,   # tile grid width and dense row stride
    batch_rows,         # number of rows in this output batch
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse       # which row-tile within batch
    k_tile = pid % grid_n_sparse                   # which column-tile

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    # Unpack uint8 bitmask → bool mask of length TILE_NUMEL.
    #   mask[i] = (bitmask[i//8] >> (i%8)) & 1
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    # rank[i] = cumulative count of set bits before position i
    # The nonzero values for this tile occupy vals[base : base + count[tile]],
    # and the i-th nonzero belongs at vals[base + rank[i]].
    offset = tl.load(layer_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
    ranks = tl.cumsum(mask_bits, 0) - 1
    value_indices = base + ranks
    v = load_15bit_at_indices(vals_ptr, value_indices, mask_bits == 1, BF16=BF16)

    v_2d = tl.reshape(v, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, v_2d, mask=(offs_m < batch_rows) & (offs_k < K))


# ═══════════════════════════════════════════════════════════════════════════════
# _unpack_values_batch_kernel
#   Reconstructs dense tiles from an ordinary signed 16-bit sparse stream.
@triton.jit
def _unpack_values_batch_kernel(
    vals_ptr,
    bitmask_ptr,
    prefix_ptr,
    vals_offset_ptr,
    dense_ptr,
    first_m_tile,
    grid_n_sparse, K,
    batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """Unpack an ordinary 16-bit tile-sparse value stream."""
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bits = (tl.reshape(bytes_val, (TILE_BYTES, 1))
            >> tl.arange(0, 8)[None, :]) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    base = tl.load(prefix_ptr + tile_id) + tl.load(vals_offset_ptr)
    ranks = tl.cumsum(mask_bits, 0) - 1
    values = tl.load(vals_ptr + base + ranks, mask=mask_bits == 1, other=0.0)
    values_2d = tl.reshape(values, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(
        dense_ptr + offs,
        values_2d,
        mask=(offs_m < batch_rows) & (offs_k < K),
    )


# ═════════════════════════════════════════════════════════════════════════════
# _unpack_relu2_batch_kernel
#   Unpacks stored r = relu(a) tiles as k * r² into dense output.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _unpack_relu2_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
    BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    offset = tl.load(vals_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
    ranks = tl.cumsum(mask_bits, 0) - 1
    value_indices = base + ranks
    r = load_15bit_at_indices(vals_ptr, value_indices, mask_bits == 1, BF16=BF16)
    #rdtype = r.dtype
    #r = r.to(tl.float32)
    z = RELU2_SCALE * r * r
    #z = z.to(rdtype)
    z_2d = tl.reshape(z, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, z_2d, mask=(offs_m < batch_rows) & (offs_k < K))


# ═══════════════════════════════════════════════════════════════════════════════
# _relu_grad_sparse_kernel
#   Computes:  grad_preact = grad * (relu(a) > 0)
#   In-place update on grad.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _relu_grad_sparse_kernel(
    grad_ptr,           # input/output: dense gradient ∂L/∂Z ∈ R^{M×N} (in-place)
    bitmask_ptr,        # input:  uint8 packed bitmasks
    M, N,               # dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    gz = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = tl.reshape((bytes_2d >> bit_pos) & 1, (BLOCK_M, BLOCK_N))

    # Element-wise:  gz[p,q] = 0 if Z[p,q] ≤ 0, else gz[p,q]
    masked = tl.where(bits != 0, gz, 0.0)
    tl.store(grad_ptr + offs, masked, mask=(rm[:, None] < M) & (rn[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════════
# _relu2_grad_sparse_kernel
#   Computes:  grad_preact = grad * 2 * k * r  (for active entries, r = relu(a) > 0)
#   where z = k * r^2, so the derivative w.r.t. the preactivation is dz/da = 2*k*r.
#   In-place update on grad.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _relu2_grad_sparse_kernel(
    grad_ptr, vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
    BF16: tl.constexpr,
):
    """In-place: grad <- grad * dz/da, where dz/da = 2*k*r for r = relu(a) > 0,
    else 0 (matching the stored bitmask)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    grad = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    mask_bits = tl.reshape((bytes_2d >> tl.arange(0, 8)[None, :]) & 1, (TILE_NUMEL,))

    offset = tl.load(vals_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
    ranks = tl.cumsum(mask_bits, 0) - 1
    value_indices = base + ranks
    r = load_15bit_at_indices(vals_ptr, value_indices, mask_bits == 1, BF16=BF16)
    r = r.to(tl.float32)
    scale = 2.0 * RELU2_SCALE * r
    scale_2d = tl.reshape(scale, (BLOCK_M, BLOCK_N))
    bits_2d = tl.reshape(mask_bits, (BLOCK_M, BLOCK_N))

    grad_preact = tl.where(bits_2d != 0, grad * scale_2d, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════
# _relu2_layer_grad_kernel
#   Computes a signed sparse dpreact without modifying saved activations.
# ═══════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "D"])
@triton.jit
def _relu2_layer_grad_kernel(
    grad_output_ptr,
    W2_ptr,
    vals_ptr,
    bitmask_ptr,
    prefix_ptr,
    vals_offset_ptr,
    vals_out_ptr,
    M, N, grid_n,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
    BF16: tl.constexpr,
):
    """Fuse ``grad_output @ W2`` with the ReLU² derivative and compaction."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_id = pid_m * grid_n + pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, D, BLOCK_K):
        k = k_start + offs_k
        grad_output = tl.load(
            grad_output_ptr + offs_m[:, None] * D + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < D),
            other=0.0,
        )
        weights = tl.load(
            W2_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < D) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(grad_output, weights)

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    mask_bits = tl.reshape(
        (tl.reshape(bytes_val, (TILE_BYTES, 1))
         >> tl.arange(0, 8)[None, :]) & 1,
        (TILE_NUMEL,),
    )

    input_base = tl.load(prefix_ptr + tile_id) + tl.load(vals_offset_ptr)
    ranks = tl.cumsum(mask_bits, 0) - 1
    value_indices = input_base + ranks
    relu = load_15bit_at_indices(vals_ptr, value_indices, mask_bits == 1, BF16=BF16)
    grad = tl.reshape(acc, (TILE_NUMEL,)) * (2.0 * RELU2_SCALE * relu)

    output_base = tl.load(prefix_ptr + tile_id)
    tl.store(vals_out_ptr + output_base + ranks, grad, mask=mask_bits == 1)


# ══════════════════════════════════════════════════════════════════════════════
# _relu_layer_grad_kernel
#   Fuses grad_output @ W2 with the saved ReLU mask and sparse compaction.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "D"])
@triton.jit
def _relu_layer_grad_kernel(
    grad_output_ptr,
    W2_ptr,
    bitmask_ptr,
    prefix_ptr,
    vals_out_ptr,
    M, N, grid_n,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_id = pid_m * grid_n + pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, D, BLOCK_K):
        k = k_start + offs_k
        grad_output = tl.load(
            grad_output_ptr + offs_m[:, None] * D + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < D),
            other=0.0,
        )
        weights = tl.load(
            W2_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < D) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(grad_output, weights)

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    mask_bits = tl.reshape(
        (tl.reshape(bytes_val, (TILE_BYTES, 1))
         >> tl.arange(0, 8)[None, :]) & 1,
        (TILE_NUMEL,),
    )
    ranks = tl.cumsum(mask_bits, 0) - 1
    values = tl.reshape(acc, (TILE_NUMEL,))
    output_base = tl.load(prefix_ptr + tile_id)
    tl.store(vals_out_ptr + output_base + ranks, values, mask=mask_bits == 1)
