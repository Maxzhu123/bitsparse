import torch
from torch import Tensor

from src.code.triton_kernels import (
    _tile_pack_kernel,
    _compact_vals_16_kernel,
    _compact_vals_staging_kernel,
    _compact_vals_15_kernel,
    _compact_vals_15_fused_kernel,
    _unpack_batch_kernel,
    _unpack_batch_15_kernel,
    _unpack_relu2_batch_kernel,
    _unpack_relu2_batch_15_kernel,
    _relu_grad_sparse_kernel,
    _relu2_grad_sparse_kernel,
    _relu2_grad_sparse_15_kernel,
)
from src.bitsparse import RELU2_SCALE, BitsparseTensor


# Fusion saves a launch for small grids, but its larger register footprint
# reduces occupancy on larger ones. This cutoff is benchmarked on the supported
# 64x64 tile format and keeps the faster implementation for each regime.
FUSED_15BIT_MAX_TILES = 1024


def tile_pack(
    dense: Tensor, tile_counts: Tensor, tile_bitmasks: Tensor,
    M: int, N: int, grid_m: int, grid_n: int,
    BLOCK_M: int, BLOCK_N: int,
    TILE_NUMEL: int, TILE_BYTES: int,
) -> None:
    """Pack dense tiles into bitmasks and nonzero counts (in-place outputs)."""
    _tile_pack_kernel[(grid_m, grid_n)](
        dense, tile_counts, tile_bitmasks,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
    )


def compact_vals(
    dense: Tensor, tile_counts: Tensor, tile_prefix: Tensor,
    vals: Tensor, vals_offset: Tensor,
    M: int, N: int, grid_n: int, num_tiles: int,
    BLOCK_M: int, BLOCK_N: int, TILE_NUMEL: int, packed_15bit: bool,
) -> None:
    """Scatter positive values into raw or packed compact storage."""
    if packed_15bit:
        raw_prefix = torch.empty(
            num_tiles + 1, device=dense.device, dtype=torch.int32
        )
        torch.cumsum(tile_counts * dense.element_size(), 0, out=raw_prefix[1:])
        raw_prefix[0] = 0
        raw_vals = torch.empty(
            raw_prefix[-1].item() // dense.element_size(),
            device=dense.device,
            dtype=dense.dtype,
        )
        if num_tiles <= FUSED_15BIT_MAX_TILES:
            _compact_vals_15_fused_kernel[(num_tiles,)](
                dense, raw_vals, raw_prefix, tile_prefix,
                vals.view(-1).view(torch.int32), vals_offset,
                M, N, grid_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                TILE_NUMEL=TILE_NUMEL,
            )
        else:
            _compact_vals_staging_kernel[(num_tiles,)](
                dense, raw_prefix, raw_vals,
                M, N, grid_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                TILE_NUMEL=TILE_NUMEL,
            )
            _compact_vals_15_kernel[(num_tiles,)](
                raw_vals, raw_prefix, tile_prefix, vals.view(-1).view(torch.int32),
                vals_offset, M, N,
            )
        return
    _compact_vals_16_kernel[(num_tiles,)](
        dense, tile_prefix, vals, vals_offset,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
    )


def unpack_batch_(
    sparse: BitsparseTensor, output: Tensor,
    first_m_tile: int, grid_n: int, K: int, batch_rows: int,
    num_tiles_in_batch: int,
) -> Tensor:
    """Unpack slice of sparse tiles into a dense output``batch_rows x K`` slice (in-place)."""
    BLOCK_M = sparse.BLOCK_M
    BLOCK_N = sparse.BLOCK_N
    if sparse.packed_15bit:
        _unpack_batch_15_kernel[(num_tiles_in_batch,)](
            sparse.vals.view(-1).view(torch.int32),
            sparse.bitmask, sparse.prefix, sparse.vals_offset,
            output,
            first_m_tile, grid_n, K, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
            IS_BF16=sparse.value_dtype == torch.bfloat16,
        )
        return output
    _unpack_batch_kernel[(num_tiles_in_batch,)](
        sparse.vals, sparse.bitmask, sparse.prefix, sparse.vals_offset,
        output,
        first_m_tile, grid_n, K, batch_rows,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
    )
    return output


def unpack_relu2_batch_(
    sparse: BitsparseTensor, output: Tensor,
    first_m_tile: int, grid_n: int, K: int, batch_rows: int,
    num_tiles_in_batch: int,
) -> Tensor:
    """Unpack stored ``r = relu(a)`` tiles as ``k * r²`` into dense (in-place)."""
    BLOCK_M = sparse.BLOCK_M
    BLOCK_N = sparse.BLOCK_N
    if sparse.packed_15bit:
        _unpack_relu2_batch_15_kernel[(num_tiles_in_batch,)](
            sparse.vals.view(-1).view(torch.int32),
            sparse.bitmask, sparse.prefix, sparse.vals_offset,
            output,
            first_m_tile, grid_n, K, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
            RELU2_SCALE=RELU2_SCALE,
            IS_BF16=sparse.value_dtype == torch.bfloat16,
        )
        return output
    _unpack_relu2_batch_kernel[(num_tiles_in_batch,)](
        sparse.vals, sparse.bitmask, sparse.prefix, sparse.vals_offset,
        output,
        first_m_tile, grid_n, K, batch_rows,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
    )
    return output


def mask_with_bitmask_(grad: Tensor, sparse: BitsparseTensor) -> Tensor:
    """Apply the saved ReLU mask in-place: ``grad <- grad * bitmask``."""
    BLOCK_M = sparse.BLOCK_M
    BLOCK_N = sparse.BLOCK_N
    _relu_grad_sparse_kernel[(sparse.grid_m, sparse.grid_n)](
        grad, sparse.bitmask,
        sparse.shape[0], sparse.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_BYTES=BLOCK_M * BLOCK_N // 8,
    )
    return grad


def relu2_grad_sparse_(grad: Tensor, sparse_z: BitsparseTensor) -> Tensor:
    """Apply the ReLU² derivative in-place on ``grad`` using sparse ``z``.

    Computes `dpreact = grad * 2 * k * r` for active entries, where
    `z = k * r²` and `r = relu(a)` is stored sparsely.  `grad` is
    overwritten with the result and returned.

    Autotuned; the kernel's restore_value resets grad between benchmarks.
    """
    BLOCK_M = sparse_z.BLOCK_M
    BLOCK_N = sparse_z.BLOCK_N
    if sparse_z.packed_15bit:
        _relu2_grad_sparse_15_kernel[(sparse_z.grid_m, sparse_z.grid_n)](
            grad, sparse_z.vals.view(-1).view(torch.int32),
            sparse_z.bitmask, sparse_z.prefix, sparse_z.vals_offset,
            sparse_z.shape[0], sparse_z.shape[1],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=BLOCK_M * BLOCK_N,
            TILE_BYTES=BLOCK_M * BLOCK_N // 8,
            RELU2_SCALE=RELU2_SCALE,
            IS_BF16=sparse_z.value_dtype == torch.bfloat16,
        )
        return grad
    _relu2_grad_sparse_kernel[(sparse_z.grid_m, sparse_z.grid_n)](
        grad, sparse_z.vals, sparse_z.bitmask, sparse_z.prefix, sparse_z.vals_offset,
        sparse_z.shape[0], sparse_z.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N,
        TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
    )
    return grad
