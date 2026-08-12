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

  prefix   — uint32 prefix sum of per-tile nonzero counts: prefix[t] is
             the logical value offset of tile t in the continuous stream.
"""

import triton
import triton.language as tl

from .bitpacking import load_15bit_at_indices

# ═══════════════════════════════════════════════════════════════════════════════
# Autotune configs for the hot kernels.
# ═══════════════════════════════════════════════════════════════════════════════
# Memory-bound gather kernels: configs are keyed on the dense output shape so
# each distinct tile-grid / batch size benchmarks once.
_UNPACK_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=1),
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

# Tile compaction is dominated by the 4096-element scan and scatter. Keep the
# search small: wider launches help the scan, while extra pipeline stages have
# limited value for this memory-bound kernel.
_COMPACT_VALS_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
]

# ══════════════════════════════════════════════════════════════════════════════
# _tile_pack_kernel
#   Computes:  bitmask[t] = pack(X_tile > 0)    ∀ tile t
#              counts[t]  = ||X_tile > 0||₀     (number of positive entries)
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=_COMPACT_VALS_CONFIGS, key=["M", "N"],
)
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
#   The raw path scatters positive values into one logical stream. The packed
#   path stages that stream as 16-bit values, then packs it continuously.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=_COMPACT_VALS_CONFIGS, key=["M", "N"],
)
@triton.jit
def _compact_vals_kernel(
    dense_ptr,          # input:  dense X ∈ R^{M×N}
    tile_prefix_ptr,    # input:  uint32[n_tiles+1] logical value offsets
    vals_out_ptr,       # output: compact value buffer for positive values
    output_offest_ptr,   # input:  int64[1] global logical value offset
    first_tile, M, N, grid_n,  # chunk start, dimensions, and tile grid
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
    quantize: tl.constexpr, scale_ptr,  # FP8 storage: divide by scale before rounding
):
    pid = tl.program_id(0)
    tile_id = first_tile + pid
    base = (tl.load(tile_prefix_ptr + tile_id)
            - tl.load(tile_prefix_ptr + first_tile)
            + tl.load(output_offest_ptr))

    tile_m = tile_id // grid_n
    tile_n = tile_id % grid_n

    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v_2d = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    v = tl.reshape(v_2d, (TILE_NUMEL,))

    nz = (v > 0.0).to(tl.int32)

    # rank[i] = number of nonzero entries before position i within this tile.
    # Used as the offset from 'base' to write the i-th nonzero value.
    ranks = tl.cumsum(nz, 0) - 1
    if quantize:
        # Promote to fp32, then round fp8.  The bitmask above is computed from
        # the original BF16 values, so positives rounded to zero keep their mask.
        v = (v.to(tl.float32) / tl.load(scale_ptr)).to(tl.float8e4nv)
    tl.store(vals_out_ptr + base + ranks, v, mask=(nz == 1))


# ═══════════════════════════════════════════════════════════════════════════════
# _unpack_batch_kernel / _unpack_relu2_batch_kernel
#   Reconstructs dense tiles from the sparse representation.
#   For each tile t in a batch of rows:
#     for each nonzero position i in tile t (from bitmask[t]):
#         D_tile[i] = vals[prefix[t] + rank[i]]
#
# vals can be packed sbit or raw.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _unpack_tile(
    vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    first_m_tile, grid_n_sparse,
    pack_sbit: tl.constexpr, fp8: tl.constexpr, scale_ptr,
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
    base = tl.load(prefix_ptr + tile_id) + tl.load(vals_offset_ptr)
    ranks = tl.cumsum(mask_bits, 0) - 1

    if pack_sbit:
        return load_15bit_at_indices(vals_ptr, base + ranks, mask_bits == 1)
    elif fp8:
        # Promote to fp32 before dequantizing so downstream math stays in fp32.
        q = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)
        return q.to(tl.float32) * tl.load(scale_ptr)
    else:
        return tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)


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


@triton.autotune(configs=_UNPACK_CONFIGS, key=["grid_n_sparse", "K", "batch_rows", "pack_sbit", "fp8", "square_vals"])
@triton.jit
def _unpack_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    square_vals: tl.constexpr, RELU2_SCALE: tl.constexpr, pack_sbit: tl.constexpr,
    fp8: tl.constexpr, scale_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """ Unpack stored tile values as-is into a dense ``[batch_rows, K]`` slice.
        Handles sign bit packing
        vals can be squared if using ReLU^2 activation.
        """
    vals = _unpack_tile(vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
                            first_m_tile, grid_n_sparse,
                            pack_sbit, fp8, scale_ptr,
                            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES)

    if square_vals:
        vals = RELU2_SCALE * vals * vals

    _store_tile(dense_ptr, vals, grid_n_sparse, batch_rows, K,
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
@triton.autotune(configs=_MASK_CONFIGS, key=["M", "N", "pack_sbit", "fp8"], restore_value=["grad_ptr"])
@triton.jit
def _relu2_grad_sparse_kernel(
    grad_ptr, vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    M, N,
    pack_sbit: tl.constexpr, fp8: tl.constexpr, scale_ptr,
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
    base = tl.load(prefix_ptr + tile_id) + tl.load(vals_offset_ptr)
    ranks = tl.cumsum(mask_bits, 0) - 1
    active = mask_bits == 1
    if pack_sbit:
        r = load_15bit_at_indices(vals_ptr, base + ranks, active)
    elif fp8:
        r = tl.load(vals_ptr + base + ranks, mask=active, other=0.0).to(tl.float32) * tl.load(scale_ptr)
    else:
        r = tl.load(vals_ptr + base + ranks, mask=active, other=0.0)

    scale_2d = tl.reshape(2.0 * RELU2_SCALE * r.to(tl.float32), (BLOCK_M, BLOCK_N))
    grad_preact = tl.where(mask_2d != 0, grad * scale_2d, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))
