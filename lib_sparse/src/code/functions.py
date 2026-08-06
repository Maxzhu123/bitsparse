from typing import NamedTuple

import torch
from torch import Tensor

from src.code.triton_operators import compact_vals, tile_pack
from src.bitsparse import BitsparseTensor, TensorBuffer, tile_grid, BLOCK_M, BLOCK_N


class _PreparedTiles(NamedTuple):
    M: int
    N: int
    grid_m: int
    grid_n: int
    num_tiles: int
    tile_numel: int
    bitmasks: Tensor
    prefix: Tensor


def _prepare_tiles(dense: Tensor) -> _PreparedTiles:
    """Create tile masks and enqueue the logical nonzero prefix sum."""
    if dense.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("BitsparseTensor supports float16 and bfloat16 values")
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    tile_pack(dense, tile_counts, tile_bitmasks,
              M, N, grid_m, grid_n, BLOCK_M, BLOCK_N, TILE_NUMEL, TILE_BYTES)

    # ``tensor[0] = 0`` invokes PyTorch's synchronizing scalar-assignment path.
    # Zero-initialize on-device so TensorBuffer conversion stays asynchronous.
    tile_prefix = torch.zeros(num_tiles + 1, device=dense.device, dtype=torch.uint32)
    # PyTorch has no uint32 cumsum; the int32 view preserves the unsigned bits.
    torch.cumsum(tile_counts, 0, out=tile_prefix.view(torch.int32)[1:])

    return _PreparedTiles(
        M, N, grid_m, grid_n, num_tiles, TILE_NUMEL,
        tile_bitmasks, tile_prefix,
    )


def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: TensorBuffer | None = None,
    packed_15bit: bool = False,
) -> BitsparseTensor:
    """Convert one dense matrix into a :class:`BitsparseTensor`."""
    prepared = _prepare_tiles(dense)
    M, N = prepared.M, prepared.N
    grid_m, grid_n = prepared.grid_m, prepared.grid_n
    num_tiles, TILE_NUMEL = prepared.num_tiles, prepared.tile_numel
    tile_bitmasks, tile_prefix = prepared.bitmasks, prepared.prefix

    if sparse_data is None:
        vals = None
        vals_offset = None
    else:
        if sparse_data.dtype != dense.dtype:
            raise ValueError("TensorBuffer dtype must match the dense tensor dtype")
        if sparse_data.packed_15bit != packed_15bit:
            raise ValueError("packed_15bit must match the TensorBuffer encoding")
        vals = sparse_data.vals
        if packed_15bit:
            # Eight logical values occupy exactly fifteen bytes. Aligning each
            # activation keeps independent pack launches from sharing bytes.
            vals_offset = ((sparse_data.offset + 7) // 8 * 8).clone()
        else:
            vals_offset = sparse_data.offset.clone()

    vals, vals_offset = compact_vals(
        dense, tile_prefix, vals, vals_offset,
        M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL,
        packed_15bit,
    )

    if sparse_data is not None:
        sparse_data.offset.copy_(vals_offset + tile_prefix[-1])

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, dense.shape,
        vals_offset=vals_offset, packed_15bit=packed_15bit,
        value_dtype=dense.dtype,
    )
