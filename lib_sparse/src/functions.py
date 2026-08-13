from typing import NamedTuple

import torch
from torch import Tensor

from .triton_operators import compact_vals, tile_pack
from ..bitsparse import BitsparseTensor, TensorBuffer, is_fp8, tile_grid
from ..config import BLOCK_M, BLOCK_N

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


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
    if dense.dtype != torch.bfloat16 and not is_fp8(dense.dtype):
        raise TypeError("BitsparseTensor supports bfloat16 and FP8 (e4m3fn/e5m2) values")
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


def _compute_scale(dense: Tensor, storage_dtype: torch.dtype) -> Tensor:
    """Per-tensor FP8 scale mapping ``max|X|`` to the largest stored value."""
    amax = dense.detach().abs().amax().float()
    scale = amax / torch.finfo(storage_dtype).max
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def dense_to_tilesparse(
    dense: Tensor, scale: Tensor|None,
    sparse_data: TensorBuffer | None = None,
    pack_sbit: bool = False,
) -> BitsparseTensor:
    """Convert dense tensor into a BitsparseTensor.

    FP8 values are compacted as-is; BF16 values are quantized to FP8 storage
    with a per-tensor scale that becomes part of the sparse metadata.  Only the
    bf16-to-fp8 path needs a scale, so fp8 inputs skip it entirely.
    Reconstructed tensors stay in the input's dtype.
    """
    prepared = _prepare_tiles(dense)
    M, N = prepared.M, prepared.N
    grid_m, grid_n = prepared.grid_m, prepared.grid_n
    num_tiles, TILE_NUMEL = prepared.num_tiles, prepared.tile_numel
    tile_bitmasks, tile_prefix = prepared.bitmasks, prepared.prefix

    if sparse_data is None:
        vals = None
        vals_offset = None
    else:
        vals = sparse_data.vals
        if pack_sbit:
            # Eight logical values occupy exactly fifteen bytes. Aligning each
            # activation keeps independent pack launches from sharing bytes.
            vals_offset = ((sparse_data.offset + 7) // 8 * 8).clone()
        else:
            vals_offset = sparse_data.offset.clone()

    vals, vals_offset = compact_vals(
        dense, tile_bitmasks, tile_prefix, vals, vals_offset,
        M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL,
        pack_sbit,
    )

    if sparse_data is not None:
        sparse_data.offset.copy_(vals_offset + tile_prefix[-1])

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, dense.shape,
        dense.dtype, scale,
        vals_offset=vals_offset, pack_sbit=pack_sbit,
    )


def to_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    scale = (x.detach().abs().max() / _FP8_MAX).to(torch.float32)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x_fp8 = (x / scale).to(_FP8_DTYPE).contiguous()
    return x_fp8, scale

