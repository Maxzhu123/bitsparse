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

  prefix   — int32 prefix sum of per-tile nonzero counts: prefix[t] is
             the starting offset of tile t's values inside vals.
             prefix[num_tiles] equals the total number of nonzero values.
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
# _compact_vals_kernel
#   Given prefix[t] = Σ_{i=0}^{t-1} count[i] (exclusive prefix sum),
#   scatters tile t's nonzero values into a global compact buffer:
#     vals[prefix[t] : prefix[t+1]] = {X[p,q] : (p,q) ∈ tile t, X[p,q] > 0}
#   Values within each tile are stored in row-major order.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=_COMPACT_CONFIGS, key=["M", "N"])
@triton.jit
def _compact_vals_kernel(
    dense_ptr,          # input:  dense X ∈ R^{M×N}
    tile_prefix_ptr,    # input:  int32[n_tiles+1] exclusive prefix sum of counts
    vals_out_ptr,       # output: compact bf16 buffer for nonzero values
    layer_offset_ptr,   # input:  int32[1] global offset where this layer starts
    M, N, grid_n,       # dimensions and tile grid
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.load(layer_offset_ptr)
    base = tl.load(tile_prefix_ptr + pid) + offset   # absolute position in vals

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
@triton.autotune(configs=_UNPACK_CONFIGS, key=["grid_n_sparse", "K", "batch_rows"])
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
    v = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)

    v_2d = tl.reshape(v, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, v_2d, mask=(offs_m < batch_rows) & (offs_k < K))


# ═══════════════════════════════════════════════════════════════════════════════
# _unpack_relu2_batch_kernel
#   Unpacks stored r = relu(a) tiles as k * r² into dense output.
# ═══════════════════════════════════════════════════════════════════════════════
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
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)
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
@triton.autotune(configs=_MASK_CONFIGS, key=["M", "N"])
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
@triton.autotune(configs=_MASK_CONFIGS, key=["M", "N"])
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

    offset = tl.load(vals_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0).to(tl.float32)
    scale = 2.0 * RELU2_SCALE * r
    scale_2d = tl.reshape(scale, (BLOCK_M, BLOCK_N))
    bits_2d = tl.reshape(mask_bits, (BLOCK_M, BLOCK_N))

    grad_preact = tl.where(bits_2d != 0, grad * scale_2d, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))

