"""
Per-tile compressed sparse format — shared kernel library.

A dense 2D tensor X ∈ R^{M×N} is partitioned into a grid of tiles, each
of shape [BLOCK_M × BLOCK_N].  Every tile is independently compressed:

  bitmask  — uint8 packed bitmask (8 bits per byte), TILE_BYTES bytes/tile.
             Row-major within the tile, so bit at flat offset f lives in
             byte f//8 at bit position f%8.  A set bit (1) means the
             element is nonzero after ReLU: X[i,j] > 0.

  vals     — a single compact 1D array containing all nonzero values
             across all tiles, concatenated in grid-major order.

  prefix   — int32 prefix sum of per-tile storage bytes: prefix[t] is
             the starting byte offset of tile t's values inside vals.
             Packed tiles are independently aligned to 32-bit words.
"""

import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════════════
# Autotune configs for the hot kernels.
#
# All kernels are autotuned over num_warps / num_stages:
#   - _tile_pack_kernel, _compact_vals_kernel (run once per layer on every
#     forward pass) and the two tl.dot matmul kernels (where the inner-loop
#     tile BLOCK_K is a pure performance parameter).
#   - The memory-bound elementwise/gather kernels (unpack_*, mask_with_bitmask,
#     relu2_grad_sparse) are also autotuned so num_warps/num_stages are picked
#     per shape instead of hand-tuned.
#
# _relu2_grad_sparse_kernel is in-place/non-idempotent, so its autotune uses
# restore_value=["grad_ptr"] to reset grad between benchmark iterations.
#
# BLOCK_M/BLOCK_N/TILE_* are NOT tuned: they define the BitsparseTensor format
# itself and every kernel operating on a tensor must agree on them.
#
# Config sets are deliberately small and keyed only on shape dims so
# each fixed layer shape benchmarks exactly once.
# ═══════════════════════════════════════════════════════════════════════════════
_PACK_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
]

_COMPACT_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=16, num_stages=2),
]

_MATMUL_CONFIGS = [
    triton.Config({"BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 32}, num_warps=8, num_stages=3),  # previous hand-tuned
    triton.Config({"BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=3),
]

# Memory-bound gather kernels: configs are keyed on the dense output shape so
# each distinct tile-grid / batch size benchmarks once.
_UNPACK_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=16, num_stages=2),
]

# Elementwise mask / gradient kernels (one tile per program, 2D grid).
_MASK_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
]

# ═══════════════════════════════════════════════════════════════════════════════
# _tile_pack_kernel
#   Computes:  bitmask[t] = pack(X_tile > 0)    ∀ tile t
#              counts[t]  = ||X_tile > 0||₀     (number of positive entries)
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=_PACK_CONFIGS, key=["M", "N"])
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
# Value compaction
#   The raw path scatters each tile's positive values directly into its byte
#   range. The packed path first performs the same compaction into temporary
#   16-bit storage, then assigns each complete 32-bit packed output word to one
#   lane. This second stage has no overlapping writes and needs no atomics.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=_COMPACT_CONFIGS, key=["M", "N"])
@triton.jit
def _compact_vals_16_kernel(
    dense_ptr,          # input:  dense X ∈ R^{M×N}
    tile_prefix_ptr,    # input:  int32[n_tiles+1] exclusive byte offsets
    vals_out_ptr,       # output: compact fp16/bf16 buffer for positive values
    layer_offset_ptr,   # input:  int64[1] global byte offset for this layer
    M, N, grid_n,       # dimensions and tile grid
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_bytes = tl.load(layer_offset_ptr)
    base_bytes = tl.load(tile_prefix_ptr + pid) + offset_bytes
    base = base_bytes // 2

    tile_m = pid // grid_n
    tile_n = pid % grid_n

    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v_2d = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    v = tl.reshape(v_2d, (TILE_NUMEL,))

    nz = (v > 0.0).to(tl.int32)

    # rank[i] = number of nonzero entries before position i within this tile.
    # Used as the offset from 'base' to write the i-th nonzero value.
    ranks = tl.cumsum(nz, 0) - 1
    tl.store(vals_out_ptr + base + ranks, v, mask=(nz == 1))


@triton.autotune(configs=_COMPACT_CONFIGS, key=["M", "N"])
@triton.jit
def _compact_vals_staging_kernel(
    dense_ptr,
    raw_prefix_ptr,
    raw_vals_ptr,
    M, N, grid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    """Compact positive values into temporary contiguous 16-bit tile storage."""
    pid = tl.program_id(0)
    base = tl.load(raw_prefix_ptr + pid) // 2

    tile_m = pid // grid_n
    tile_n = pid % grid_n
    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v_2d = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    v = tl.reshape(v_2d, (TILE_NUMEL,))

    nz = (v > 0.0).to(tl.int32)
    ranks = tl.cumsum(nz, 0) - 1
    tl.store(raw_vals_ptr + base + ranks, v, mask=nz == 1)


@triton.autotune(configs=_COMPACT_CONFIGS, key=["M", "N"])
@triton.jit
def _compact_vals_15_kernel(
    raw_vals_ptr,
    raw_prefix_ptr,
    tile_prefix_ptr,
    vals_words_ptr,
    layer_offset_ptr,
    M, N,
):
    """Pack contiguous 16-bit tile values into aligned 15-bit output words."""
    pid = tl.program_id(0)
    raw_start_bytes = tl.load(raw_prefix_ptr + pid)
    raw_end_bytes = tl.load(raw_prefix_ptr + pid + 1)
    raw_base = raw_start_bytes // 2
    num_values = (raw_end_bytes - raw_start_bytes) // 2

    offset_bytes = tl.load(layer_offset_ptr)
    tile_start_bytes = tl.load(tile_prefix_ptr + pid)
    tile_end_bytes = tl.load(tile_prefix_ptr + pid + 1)
    base_word = (offset_bytes + tile_start_bytes) // 4
    num_words = (tile_end_bytes - tile_start_bytes) // 4

    # One lane owns each complete output word, so no writes overlap.
    word_offs = tl.arange(0, 2048)
    stream_bit = word_offs * 32
    value_idx = stream_bit // 15
    shift = stream_bit % 15

    v0 = tl.load(raw_vals_ptr + raw_base + value_idx,
                 mask=value_idx < num_values, other=0.0)
    v1 = tl.load(raw_vals_ptr + raw_base + value_idx + 1,
                 mask=(value_idx + 1) < num_values, other=0.0)
    v2 = tl.load(raw_vals_ptr + raw_base + value_idx + 2,
                 mask=(value_idx + 2) < num_values, other=0.0)
    v3 = tl.load(raw_vals_ptr + raw_base + value_idx + 3,
                 mask=(value_idx + 3) < num_values, other=0.0)

    b0 = v0.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    b1 = v1.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    b2 = v2.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    b3 = v3.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    # offset is at most 14. Three values fill the word except at offset 14,
    # where the low bit of a fourth value becomes bit 31.
    packed_word = (
        (b0 >> shift)
        | (b1 << (15 - shift))
        | (b2 << (30 - shift))
        | tl.where(shift == 14, b3 << 31, 0)
    )

    tl.store(vals_words_ptr + base_word + word_offs, packed_word,
             mask=word_offs < num_words)


@triton.autotune(configs=_COMPACT_CONFIGS, key=["M", "N"])
@triton.jit
def _compact_vals_15_fused_kernel(
    dense_ptr,
    raw_vals_ptr,
    raw_prefix_ptr,
    tile_prefix_ptr,
    vals_words_ptr,
    layer_offset_ptr,
    M, N, grid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    """Compact to temporary 16-bit storage and pack it in one tile program."""
    pid = tl.program_id(0)
    raw_start_bytes = tl.load(raw_prefix_ptr + pid)
    raw_end_bytes = tl.load(raw_prefix_ptr + pid + 1)
    raw_base = raw_start_bytes // 2
    num_values = (raw_end_bytes - raw_start_bytes) // 2

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
    tl.store(raw_vals_ptr + raw_base + ranks, v, mask=nz == 1)

    # Phase two reads only staging values written by this same tile program.
    tl.debug_barrier()

    offset_bytes = tl.load(layer_offset_ptr)
    tile_start_bytes = tl.load(tile_prefix_ptr + pid)
    tile_end_bytes = tl.load(tile_prefix_ptr + pid + 1)
    base_word = (offset_bytes + tile_start_bytes) // 4
    num_words = (tile_end_bytes - tile_start_bytes) // 4

    word_offs = tl.arange(0, 2048)
    stream_bit = word_offs * 32
    value_idx = stream_bit // 15
    shift = stream_bit % 15
    v0 = tl.load(raw_vals_ptr + raw_base + value_idx,
                 mask=value_idx < num_values, other=0.0)
    v1 = tl.load(raw_vals_ptr + raw_base + value_idx + 1,
                 mask=(value_idx + 1) < num_values, other=0.0)
    v2 = tl.load(raw_vals_ptr + raw_base + value_idx + 2,
                 mask=(value_idx + 2) < num_values, other=0.0)
    v3 = tl.load(raw_vals_ptr + raw_base + value_idx + 3,
                 mask=(value_idx + 3) < num_values, other=0.0)

    b0 = v0.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    b1 = v1.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    b2 = v2.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    b3 = v3.to(tl.uint16, bitcast=True).to(tl.uint32) & 0x7FFF
    packed_word = (
        (b0 >> shift)
        | (b1 << (15 - shift))
        | (b2 << (30 - shift))
        | tl.where(shift == 14, b3 << 31, 0)
    )
    tl.store(vals_words_ptr + base_word + word_offs, packed_word,
             mask=word_offs < num_words)


# ═══════════════════════════════════════════════════════════════════════════════
# _unpack_batch_kernel / _unpack_relu2_batch_kernel
#   Reconstructs dense tiles from the sparse representation.
#   For each tile t in a batch of rows:
#     D_tile = 0
#     for each nonzero position i in tile t (from bitmask[t]):
#         D_tile[i] = vals[prefix[t] + rank[i]]
#   This computes: D_rowslice = gather(vals, bitmask, prefix)
#   where D_rowslice ∈ R^{batch_rows × K} is written into dense_ptr.
#
#   Both kernels share the same gather + store; the only difference is the
#   elementwise transform applied before writing: identity for _unpack_batch_kernel
#   vs ``r → k * r²`` for _unpack_relu2_batch_kernel.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _unpack_tile_16(
    vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    first_m_tile, grid_n_sparse,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """Gather this program's tile values from the compact store."""
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    tile_id = (first_m_tile + row_tile_in_batch) * grid_n_sparse + k_tile

    # Unpack uint8 bitmask → bool mask of length TILE_NUMEL.
    #   mask[i] = (bitmask[i//8] >> (i%8)) & 1
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    # rank[i] = cumulative count of set bits before position i; the i-th
    # nonzero value sits at vals[base + rank[i]].
    offset_bytes = tl.load(vals_offset_ptr)
    base = (tl.load(prefix_ptr + tile_id) + offset_bytes) // 2
    ranks = tl.cumsum(mask_bits, 0) - 1
    return tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)


@triton.jit
def _unpack_tile_15(
    vals_words_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    first_m_tile, grid_n_sparse,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """Gather this program's values from a word-aligned 15-bit tile stream."""
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse
    tile_id = (first_m_tile + row_tile_in_batch) * grid_n_sparse + k_tile

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos_in_byte = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos_in_byte) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    offset_bytes = tl.load(vals_offset_ptr)
    tile_start_bytes = tl.load(prefix_ptr + tile_id)
    tile_end_bytes = tl.load(prefix_ptr + tile_id + 1)
    base_word = (offset_bytes + tile_start_bytes) // 4
    num_words = (tile_end_bytes - tile_start_bytes) // 4

    ranks = tl.cumsum(mask_bits, 0) - 1
    packed_bit_pos = ranks * 15
    word_idx = packed_bit_pos // 32
    shift = packed_bit_pos % 32
    active = mask_bits == 1
    word0 = tl.load(vals_words_ptr + base_word + word_idx, mask=active, other=0).to(tl.uint32)
    word1 = tl.load(
        vals_words_ptr + base_word + word_idx + 1,
        mask=active & ((word_idx + 1) < num_words), other=0,
    ).to(tl.uint32)
    upper = tl.where(shift == 0, 0, word1 << (32 - shift))
    restored = ((word0 >> shift) | upper) & 0x7FFF
    return restored.to(tl.uint16)


@triton.jit
def _store_tile(
    dense_ptr, tile_vals,
    grid_n_sparse, batch_rows, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Write one reshaped tile into the dense ``[batch_rows, K]`` output."""
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    v_2d = tl.reshape(tile_vals, (BLOCK_M, BLOCK_N))
    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, v_2d, mask=(offs_m < batch_rows) & (offs_k < K))


@triton.autotune(configs=_UNPACK_CONFIGS, key=["grid_n_sparse", "K", "batch_rows"])
@triton.jit
def _unpack_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """Unpack stored tile values as-is into a dense ``[batch_rows, K]`` slice."""
    vals = _unpack_tile_16(vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
                        first_m_tile, grid_n_sparse,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES)
    _store_tile(dense_ptr, vals, grid_n_sparse, batch_rows, K,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


@triton.autotune(configs=_UNPACK_CONFIGS, key=["grid_n_sparse", "K", "batch_rows"])
@triton.jit
def _unpack_relu2_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
):
    """Unpack stored ``r = relu(a)`` tiles as ``k * r²`` into dense output."""
    r = _unpack_tile_16(vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
                     first_m_tile, grid_n_sparse,
                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                     TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES)
    _store_tile(dense_ptr, RELU2_SCALE * r * r, grid_n_sparse, batch_rows, K,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


@triton.autotune(configs=_UNPACK_CONFIGS, key=["grid_n_sparse", "K", "batch_rows"])
@triton.jit
def _unpack_batch_15_kernel(
    vals_words_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    bits = _unpack_tile_15(
        vals_words_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
        first_m_tile, grid_n_sparse,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
    )
    if IS_BF16:
        vals = bits.to(tl.bfloat16, bitcast=True)
    else:
        vals = bits.to(tl.float16, bitcast=True)
    _store_tile(dense_ptr, vals, grid_n_sparse, batch_rows, K,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


@triton.autotune(configs=_UNPACK_CONFIGS, key=["grid_n_sparse", "K", "batch_rows"])
@triton.jit
def _unpack_relu2_batch_15_kernel(
    vals_words_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr, IS_BF16: tl.constexpr,
):
    bits = _unpack_tile_15(
        vals_words_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
        first_m_tile, grid_n_sparse,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
    )
    if IS_BF16:
        r = bits.to(tl.bfloat16, bitcast=True)
    else:
        r = bits.to(tl.float16, bitcast=True)
    _store_tile(dense_ptr, RELU2_SCALE * r * r, grid_n_sparse, batch_rows, K,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


# ═══════════════════════════════════════════════════════════════════════════════
# _relu_grad_sparse_kernel
#   Computes:  grad_preact = grad * (relu(a) > 0)
#   In-place update on grad.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=_MASK_CONFIGS, key=["M", "N"])
@triton.jit
def _relu_grad_sparse_kernel(
    grad_ptr,           # input/output: dense gradient ∂L/∂Z ∈ R^{M×N} (in-place)
    bitmask_ptr,        # input:  uint8 packed bitmasks
    M, N,               # dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    """In-place: grad <- grad * (relu(a) > 0), matching the stored bitmask."""
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
    mask_2d = tl.reshape((bytes_2d >> tl.arange(0, 8)[None, :]) & 1, (BLOCK_M, BLOCK_N))

    # Element-wise:  grad[p,q] = 0 if Z[p,q] ≤ 0, else grad[p,q]
    grad_preact = tl.where(mask_2d != 0, grad, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════════
# _relu2_grad_sparse_kernel
#   Computes:  grad_preact = grad * 2 * k * r  (for active entries, r = relu(a) > 0)
#   where z = k * r^2, so the derivative w.r.t. the preactivation is dz/da = 2*k*r.
#   In-place update on grad.
#
#   Autotuned with restore_value=["grad_ptr"]: resets in-place grad between
#   benchmark iterations so the non-idempotent transform never compounds.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=_MASK_CONFIGS, key=["M", "N"], restore_value=["grad_ptr"])
@triton.jit
def _relu2_grad_sparse_kernel(
    grad_ptr, vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
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
    mask_2d = tl.reshape(mask_bits, (BLOCK_M, BLOCK_N))

    # r = relu(a) gathered from the compact store; dz/da = 2*k*r.
    offset_bytes = tl.load(vals_offset_ptr)
    base = (tl.load(prefix_ptr + tile_id) + offset_bytes) // 2
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0).to(tl.float32)
    scale_2d = tl.reshape(2.0 * RELU2_SCALE * r, (BLOCK_M, BLOCK_N))

    grad_preact = tl.where(mask_2d != 0, grad * scale_2d, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.autotune(configs=_MASK_CONFIGS, key=["M", "N"], restore_value=["grad_ptr"])
@triton.jit
def _relu2_grad_sparse_15_kernel(
    grad_ptr, vals_words_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr, IS_BF16: tl.constexpr,
):
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
    mask_2d = tl.reshape(mask_bits, (BLOCK_M, BLOCK_N))

    offset_bytes = tl.load(vals_offset_ptr)
    tile_start_bytes = tl.load(prefix_ptr + tile_id)
    tile_end_bytes = tl.load(prefix_ptr + tile_id + 1)
    base_word = (offset_bytes + tile_start_bytes) // 4
    num_words = (tile_end_bytes - tile_start_bytes) // 4
    ranks = tl.cumsum(mask_bits, 0) - 1
    packed_bit_pos = ranks * 15
    word_idx = packed_bit_pos // 32
    shift = packed_bit_pos % 32
    active = mask_bits == 1
    word0 = tl.load(vals_words_ptr + base_word + word_idx, mask=active, other=0).to(tl.uint32)
    word1 = tl.load(
        vals_words_ptr + base_word + word_idx + 1,
        mask=active & ((word_idx + 1) < num_words), other=0,
    ).to(tl.uint32)
    upper = tl.where(shift == 0, 0, word1 << (32 - shift))
    restored = (((word0 >> shift) | upper) & 0x7FFF).to(tl.uint16)
    if IS_BF16:
        r = restored.to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        r = restored.to(tl.float16, bitcast=True).to(tl.float32)

    scale_2d = tl.reshape(2.0 * RELU2_SCALE * r, (BLOCK_M, BLOCK_N))
    grad_preact = tl.where(mask_2d != 0, grad * scale_2d, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))
