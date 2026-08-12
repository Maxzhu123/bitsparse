from typing import NamedTuple

import torch
from torch import Tensor

from .triton_operators import compact_vals, tile_pack
from ..bitsparse import BitsparseTensor, TensorBuffer, tile_grid
from config import BLOCK_M, BLOCK_N


_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


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
    if dense.dtype != torch.bfloat16:
        raise TypeError("BitsparseTensor supports bfloat16 values")
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


def _compute_scale(dense: Tensor) -> Tensor:
    """Per-tensor FP8 scale mapping ``max|X|`` to the largest e4m3 value."""
    amax = dense.detach().abs().amax()
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def _device_matches(a, b) -> bool:
    """True when both refer to the same device, tolerating ``cuda`` vs ``cuda:0``."""
    a, b = torch.device(a), torch.device(b)
    return a.type == b.type and (
        a.index is None or b.index is None or a.index == b.index
    )


def _validate_sparse_buffer(
    buffer: TensorBuffer, device, storage_dtype: torch.dtype, pack_sbit: bool,
) -> None:
    """Reject buffer/format mismatches before writing values.

    Capacity is deliberately not checked: the shared-buffer design trusts the
    caller to size for the worst case and keeps all writes asynchronous.
    """
    if buffer.dtype != storage_dtype:
        raise ValueError(
            f"TensorBuffer dtype {buffer.dtype} does not match storage "
            f"dtype {storage_dtype}"
        )
    if not _device_matches(buffer.device, device):
        raise ValueError(
            f"TensorBuffer device {buffer.device} does not match dense device {device}"
        )
    if buffer.pack_sbit != pack_sbit:
        raise ValueError(
            f"TensorBuffer pack_sbit={buffer.pack_sbit} does not match requested "
            f"pack_sbit={pack_sbit}"
        )


def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: TensorBuffer | None = None,
    pack_sbit: bool = False,
    storage_dtype: torch.dtype | None = None,
) -> BitsparseTensor:
    """Convert dense tensor into a BitsparseTensor.

    BF16 values are compacted into the requested ``storage_dtype``.  FP8
    storage quantizes with a per-tensor scale, so the scale becomes part of the
    sparse metadata; reconstructed tensors stay in the input's BF16 dtype.
    """
    if storage_dtype is None:
        storage_dtype = sparse_data.dtype if sparse_data is not None else torch.bfloat16
    if pack_sbit and storage_dtype != torch.bfloat16:
        raise ValueError(
            "pack_sbit uses the BF16-specific 15-bit codec and cannot be "
            f"combined with storage dtype {storage_dtype}"
        )

    scale = _compute_scale(dense) if storage_dtype == _FP8_DTYPE else None
    prepared = _prepare_tiles(dense)
    M, N = prepared.M, prepared.N
    grid_m, grid_n = prepared.grid_m, prepared.grid_n
    num_tiles, TILE_NUMEL = prepared.num_tiles, prepared.tile_numel
    tile_bitmasks, tile_prefix = prepared.bitmasks, prepared.prefix

    if sparse_data is None:
        vals = None
        vals_offset = None
    else:
        _validate_sparse_buffer(
            sparse_data, dense.device, storage_dtype, pack_sbit
        )
        vals = sparse_data.vals
        if pack_sbit:
            # Eight logical values occupy exactly fifteen bytes. Aligning each
            # activation keeps independent pack launches from sharing bytes.
            vals_offset = ((sparse_data.offset + 7) // 8 * 8).clone()
        else:
            vals_offset = sparse_data.offset.clone()

    vals, vals_offset = compact_vals(
        dense, tile_prefix, vals, vals_offset,
        M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL,
        pack_sbit, storage_dtype, scale,
    )

    if sparse_data is not None:
        sparse_data.offset.copy_(vals_offset + tile_prefix[-1])

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, dense.shape,
        vals_offset=vals_offset, pack_sbit=pack_sbit,
        value_dtype=storage_dtype, output_dtype=dense.dtype, scale=scale,
    )
