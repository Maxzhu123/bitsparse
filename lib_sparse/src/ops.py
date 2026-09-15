"""Opaque standalone sparse conversion for the ReLU and ReLU² autograd paths.

Capturing variable-sized storage requires callers to enable
torch._dynamo.config.capture_dynamic_output_shape_ops.
"""
import torch
from torch import Tensor

from .functions import dense_to_tilesparse as _dense_to_tilesparse
from ..bitsparse import BitsparseTensor, TensorBuffer, tile_grid
from ..config import BLOCK_M, BLOCK_N


@torch.library.custom_op(
    "bitsparse::dense_to_tilesparse",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.dynamic_output_shape, torch.Tag.cudagraph_unsafe),
)
def dense_to_tilesparse_op(
    dense: Tensor, pack_sbit: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Keep the host sync and exact-sized allocation inside the opaque op.
    sparse = _dense_to_tilesparse(dense, None, None, pack_sbit)
    return sparse.vals, sparse.bitmask, sparse.prefix, sparse.vals_offset


@dense_to_tilesparse_op.register_fake
def _fake(dense, pack_sbit=False):
    # Bind the symbolic size directly to the returned storage length. For packed
    # values it counts bytes (including padding/guard), rather than logical nnz.
    size = torch.library.get_ctx().new_dynamic_size(min=4 if pack_sbit else 0)
    dtype = torch.uint8 if pack_sbit else dense.dtype
    _, _, tiles, _, tile_bytes = tile_grid(*dense.shape, BLOCK_M, BLOCK_N)
    return (
        dense.new_empty((size,), dtype=dtype),
        dense.new_empty((tiles * tile_bytes,), dtype=torch.uint8),
        dense.new_empty((tiles + 1,), dtype=torch.uint32),
        dense.new_empty((), dtype=torch.int64),
    )


def dense_to_tilesparse_custom(
    dense: Tensor, scale: Tensor | None,
    sparse_data: TensorBuffer | None = None, pack_sbit: bool = False,
) -> BitsparseTensor:
    # Shared-buffer mutation is deliberately left on the original path for now.
    if sparse_data is not None:
        return _dense_to_tilesparse(dense, scale, sparse_data, pack_sbit)
    vals, bitmask, prefix, offset = dense_to_tilesparse_op(dense, pack_sbit)
    grid_m, grid_n, _, _, _ = tile_grid(*dense.shape, BLOCK_M, BLOCK_N)
    return BitsparseTensor(
        vals, bitmask, prefix, dense.shape, dense.dtype,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        scale=scale, vals_offset=offset, pack_sbit=pack_sbit,
    )
