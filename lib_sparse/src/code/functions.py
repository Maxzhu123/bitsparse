import torch
from torch import Tensor

from src.code.triton_operators import compact_vals, tile_pack
from src.bitsparse import BitsparseTensor, TensorBuffer, tile_grid, BLOCK_M, BLOCK_N


def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: TensorBuffer | None = None,
) -> BitsparseTensor:
    """Convert a dense activation matrix into a BitsparseTensor.

    When sparse_data is provided, values are appended to its shared buffer.
    Otherwise, this allocates a compact values tensor for this sparse tensor.
    """
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    tile_pack(dense, tile_counts, tile_bitmasks,
              M, N, grid_m, grid_n, BLOCK_M, BLOCK_N, TILE_NUMEL, TILE_BYTES)

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    if sparse_data is None:
        vals = torch.empty(tile_prefix[-1].item(), device=dense.device, dtype=dense.dtype)
        vals_offset = torch.tensor(0, device=dense.device, dtype=torch.int32)
        update_offset = None
    else:
        vals = sparse_data.vals
        vals_offset = sparse_data.offset.clone()
        update_offset = sparse_data.offset

    compact_vals(dense, tile_prefix, vals, vals_offset,
                 M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL)

    if update_offset is not None:
        update_offset.add_(tile_prefix[-1])

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, dense.shape,
        vals_offset=vals_offset,
    )
