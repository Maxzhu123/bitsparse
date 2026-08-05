import torch
from torch import Tensor

from src.code.triton_operators import compact_vals, tile_pack
from src.bitsparse import BitsparseTensor, TensorBuffer, tile_grid, BLOCK_M, BLOCK_N


def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: TensorBuffer | None = None,
    packed_15bit: bool | None = None,
) -> BitsparseTensor:
    """Convert a dense activation matrix into a BitsparseTensor.

    When sparse_data is provided, values are appended to its shared buffer.
    Otherwise, this allocates a compact values tensor for this sparse tensor.
    Set ``packed_15bit`` to remove the known-zero sign bit from positive
    FP16/BF16 values; when omitted, a supplied buffer selects the encoding.
    """
    if dense.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("BitsparseTensor supports float16 and bfloat16 values")
    if packed_15bit is None:
        packed_15bit = sparse_data.packed_15bit if sparse_data is not None else False
    elif sparse_data is not None and packed_15bit != sparse_data.packed_15bit:
        raise ValueError("packed_15bit must match the TensorBuffer encoding")

    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    tile_pack(dense, tile_counts, tile_bitmasks,
              M, N, grid_m, grid_n, BLOCK_M, BLOCK_N, TILE_NUMEL, TILE_BYTES)

    # Prefixes and buffer offsets always use bytes. Packed tiles are rounded to
    # whole 32-bit words so independently launched tile programs never share a
    # destination word.
    if packed_15bit:
        tile_storage_bytes = torch.div(
            tile_counts * 15 + 31, 32, rounding_mode="floor"
        ).mul_(4)
    else:
        tile_storage_bytes = tile_counts.mul(dense.element_size())

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_storage_bytes, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    if sparse_data is None:
        storage_bytes = tile_prefix[-1].item()
        if packed_15bit:
            vals = torch.empty(storage_bytes, device=dense.device, dtype=torch.uint8)
        else:
            vals = torch.empty(
                storage_bytes // dense.element_size(),
                device=dense.device,
                dtype=dense.dtype,
            )
        vals_offset = torch.tensor(0, device=dense.device, dtype=torch.int64)
        update_offset = None
    else:
        if sparse_data.dtype != dense.dtype:
            raise ValueError("TensorBuffer dtype must match the dense tensor dtype")
        vals = sparse_data.vals
        vals_offset = sparse_data.offset.clone()
        update_offset = sparse_data.offset

    compact_vals(dense, tile_counts, tile_prefix, vals, vals_offset,
                 M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL,
                 packed_15bit)

    if update_offset is not None:
        update_offset.add_(tile_prefix[-1])

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, dense.shape,
        vals_offset=vals_offset, packed_15bit=packed_15bit,
        value_dtype=dense.dtype,
    )
