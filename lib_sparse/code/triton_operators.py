import torch
import triton
from torch import Tensor

from .triton_kernels import (
    _tile_pack_kernel,
    _compact_vals_16_kernel,
    _unpack_batch_kernel,
    _unpack_batch_15_kernel,
    _unpack_relu2_batch_kernel,
    _unpack_relu2_batch_15_kernel,
    _relu_grad_sparse_kernel,
    _relu2_grad_sparse_kernel,
    _relu2_grad_sparse_15_kernel,
)
from ..bitsparse import RELU2_SCALE, BitsparseTensor
from .bitpacking import (
    _pack_15bit_kernel,
    packed_nbytes,
    packed_storage_nbytes,
)


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
        num_warps=2, num_stages=1,
    )


def compact_vals(
    dense: Tensor, tile_prefix: Tensor,
    vals: Tensor | None, vals_offset: Tensor | None,
    M: int, N: int, grid_n: int, num_tiles: int,
    BLOCK_M: int, BLOCK_N: int, TILE_NUMEL: int, packed_15bit: bool,
) -> tuple[Tensor, Tensor]:
    """Compact positive values into standalone or preallocated storage."""
    staging_numel = dense.numel()
    if vals is None:
        nnz = int(tile_prefix[-1].item())
        staging_numel = nnz
        size = packed_storage_nbytes(nnz) if packed_15bit else nnz
        dtype = torch.uint8 if packed_15bit else dense.dtype
        vals = torch.empty(size, device=dense.device, dtype=dtype)
        vals_offset = torch.zeros((), device=dense.device, dtype=torch.int64)

    if packed_15bit:
        # Preallocated storage uses a dense upper bound to avoid a host read.
        raw_vals = torch.empty(staging_numel, device=dense.device, dtype=dense.dtype)
        # prefix[0] supplies the zero staging offset.
        _compact_vals_16_kernel[(num_tiles,)](
            dense, tile_prefix, raw_vals, tile_prefix,
            M, N, grid_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL,
        )
        launch_bytes = packed_nbytes(staging_numel)
        if launch_bytes:
            _pack_15bit_kernel[
                lambda meta: (triton.cdiv(launch_bytes, meta["BLOCK_SIZE"]),)
            ](
                raw_vals.view(torch.uint16), vals, vals_offset,
                tile_prefix, num_tiles,
            )
        return vals, vals_offset
    _compact_vals_16_kernel[(num_tiles,)](
        dense, tile_prefix, vals, vals_offset,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
    )
    return vals, vals_offset


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
            sparse.vals.view(-1).view(torch.uint16),
            sparse.bitmask, sparse.prefix, sparse.vals_offset,
            output,
            first_m_tile, grid_n, K, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
            IS_BF16=sparse.value_dtype == torch.bfloat16,
            num_warps=4, num_stages=1,
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
            sparse.vals.view(-1).view(torch.uint16),
            sparse.bitmask, sparse.prefix, sparse.vals_offset,
            output,
            first_m_tile, grid_n, K, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
            RELU2_SCALE=RELU2_SCALE,
            IS_BF16=sparse.value_dtype == torch.bfloat16,
            num_warps=4, num_stages=1,
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
            grad, sparse_z.vals.view(-1).view(torch.uint16),
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
