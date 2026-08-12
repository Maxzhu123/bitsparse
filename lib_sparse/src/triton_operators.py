import torch
import triton
from torch import Tensor

from .triton_kernels import (
    _tile_pack_kernel,
    _compact_vals_kernel,
    _unpack_batch_kernel,
    _relu_grad_sparse_kernel,
    _relu2_grad_sparse_kernel,
)
from ..bitsparse import BitsparseTensor
from config import RELU2_SCALE, _PACK_15BIT_CHUNK_TILES
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
    )


def compact_vals(
    dense: Tensor, tile_prefix: Tensor,
    vals: Tensor | None, vals_offset: Tensor | None,
    M: int, N: int, grid_n: int, num_tiles: int,
    BLOCK_M: int, BLOCK_N: int, TILE_NUMEL: int, pack_sbit: bool,
    storage_dtype: torch.dtype = torch.bfloat16,
    scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """ Compact positive values into standalone or preallocated storage.
        Supports 15-bit packing, FP8 quantization, and preallocated vals buffer. """
    quantize = storage_dtype == torch.float8_e4m3fn
    staging_numel = dense.numel()
    standalone = vals is None
    if standalone:
        nnz = int(tile_prefix[-1].item())
        staging_numel = nnz
        size = packed_storage_nbytes(nnz) if pack_sbit else nnz
        dtype = torch.uint8 if pack_sbit else storage_dtype
        vals = torch.empty(size, device=dense.device, dtype=dtype)
        vals_offset = torch.zeros((), device=dense.device, dtype=torch.int64)

    if pack_sbit:
        chunk_tiles = min(num_tiles, _PACK_15BIT_CHUNK_TILES)
        chunk_numels = None
        if standalone:
            # Prefix boundaries give exact chunk sizes after the sync above.
            prefix_ends = [0]
            if chunk_tiles < num_tiles:
                prefix_ends.extend(
                    tile_prefix[chunk_tiles:num_tiles:chunk_tiles].tolist()
                )
            prefix_ends.append(staging_numel)
            chunk_numels = [
                end - start for start, end in zip(prefix_ends, prefix_ends[1:])
            ]
            # Reuse one workspace sized for the largest actual chunk.
            workspace_numel = max(chunk_numels)
        else:
            # Keep preallocated buffers asynchronous with a dense upper bound.
            workspace_numel = min(staging_numel, chunk_tiles * TILE_NUMEL)
        raw_vals = torch.empty(workspace_numel, device=dense.device, dtype=dense.dtype)

        for chunk_index, first_tile in enumerate(range(0, num_tiles, chunk_tiles)):
            tiles_in_chunk = min(chunk_tiles, num_tiles - first_tile)
            # Compact to workspace offset zero, then append to the packed stream.
            _compact_vals_kernel[(tiles_in_chunk,)](
                dense, tile_prefix, raw_vals, tile_prefix,
                first_tile, M, N, grid_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                TILE_NUMEL=TILE_NUMEL,
                quantize=quantize, scale_ptr=scale,
            )
            launch_numel = (
                chunk_numels[chunk_index]
                if chunk_numels is not None
                else tiles_in_chunk * TILE_NUMEL
            )
            launch_bytes = packed_nbytes(launch_numel) + 1
            _pack_15bit_kernel[
                lambda meta: (triton.cdiv(launch_bytes, meta["BLOCK_SIZE"]),)
            ](
                raw_vals.view(torch.uint16), vals, vals_offset,
                tile_prefix, first_tile, tiles_in_chunk,
            )
        return vals, vals_offset

    # Normal version version
    _compact_vals_kernel[(num_tiles,)](
        dense, tile_prefix, vals, vals_offset,
        0, M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        quantize=quantize, scale_ptr=scale,
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

    vals = sparse.vals.view(torch.uint16) if sparse.pack_sbit else sparse.vals

    _unpack_batch_kernel[(num_tiles_in_batch,)](
        vals, sparse.bitmask, sparse.prefix, sparse.vals_offset,
        output,
        first_m_tile, grid_n, K, batch_rows,
        False, 0, sparse.pack_sbit, sparse.fp8, sparse.scale,
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

    vals = sparse.vals.view(torch.uint16) if sparse.pack_sbit else sparse.vals

    _unpack_batch_kernel[(num_tiles_in_batch,)](
        vals, sparse.bitmask, sparse.prefix, sparse.vals_offset,
        output,
        first_m_tile, grid_n, K, batch_rows,
        True, RELU2_SCALE, sparse.pack_sbit, sparse.fp8, sparse.scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
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
    vals = sparse_z.vals.view(torch.uint16) if sparse_z.pack_sbit else sparse_z.vals

    _relu2_grad_sparse_kernel[(sparse_z.grid_m, sparse_z.grid_n)](
        grad, vals, sparse_z.bitmask, sparse_z.prefix, sparse_z.vals_offset,
        sparse_z.shape[0], sparse_z.shape[1],
        sparse_z.pack_sbit, sparse_z.fp8, sparse_z.scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N,
        TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
    )
    return grad
