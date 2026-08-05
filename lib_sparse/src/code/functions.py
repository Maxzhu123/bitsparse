from typing import NamedTuple

import torch
from torch import Tensor

from src.code.triton_operators import (
    compact_vals,
    compact_vals_15bit,
    mask_with_bitmask_,
    relu_layer_grad,
    relu2_grad_sparse_,
    relu2_layer_grad,
    tile_pack,
)
from src.code.sparse_matmul import (
    AspB,
    AspB_block,
    AspRelu2B_block,
    AspRelu2B,
    spAB_block,
)
from src.bitsparse import BitsparseTensor, TensorBuffer, inplace_mm_, tile_grid, BLOCK_M, BLOCK_N
from src.code.bitpacking import pack_15bit_into, packed_storage_nbytes


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
    """Create tile metadata and enqueue the count prefix sum."""
    M, N = dense.shape
    if dense.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("15-bit sparse values require a bfloat16 or float16 input")
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    tile_pack(dense, tile_counts, tile_bitmasks,
              M, N, grid_m, grid_n, BLOCK_M, BLOCK_N, TILE_NUMEL, TILE_BYTES)

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    return _PreparedTiles(
        M, N, grid_m, grid_n, num_tiles, TILE_NUMEL,
        tile_bitmasks, tile_prefix,
    )


def _finish_tilesparse(
    dense: Tensor,
    prepared: _PreparedTiles,
    sparse_data: TensorBuffer | None,
    nnz: int | None = None,
) -> BitsparseTensor:
    """Allocate or select value storage and finish a prepared conversion."""
    M, N = prepared.M, prepared.N
    grid_m, grid_n = prepared.grid_m, prepared.grid_n
    num_tiles, TILE_NUMEL = prepared.num_tiles, prepared.tile_numel
    tile_bitmasks, tile_prefix = prepared.bitmasks, prepared.prefix

    if sparse_data is None:
        if nnz is None:
            raise ValueError("nnz is required for standalone sparse storage")
        compact_values = torch.empty(nnz, device=dense.device, dtype=dense.dtype)
        vals = torch.empty(
            packed_storage_nbytes(nnz), device=dense.device, dtype=torch.uint8
        )
        vals_offset = torch.zeros(1, device=dense.device, dtype=torch.int32)
        update_offset = None
    else:
        if sparse_data.dtype != dense.dtype:
            raise TypeError(
                f"TensorBuffer dtype {sparse_data.dtype} does not match input {dense.dtype}"
            )
        vals = sparse_data.vals
        vals_offset = sparse_data.offset.clone()
        update_offset = sparse_data.offset

    if sparse_data is None:
        compact_vals(dense, tile_prefix, compact_values, vals_offset,
                     M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL)
        pack_15bit_into(compact_values, vals)
    else:
        compact_vals_15bit(dense, tile_prefix, vals, vals_offset,
                           M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL)

    if update_offset is not None:
        update_offset.add_(tile_prefix[-1])

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, dense.shape,
        vals_offset=vals_offset, dtype=dense.dtype,
    )


def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: TensorBuffer | None = None,
) -> BitsparseTensor:
    """Convert one dense matrix into a :class:`BitsparseTensor`.

    When ``sparse_data`` is provided, values are appended to its shared buffer.
    Otherwise, compact storage is allocated for this tensor.
    """
    prepared = _prepare_tiles(dense)
    nnz = None if sparse_data is not None else int(prepared.prefix[-1].item())
    return _finish_tilesparse(dense, prepared, sparse_data, nnz)


def dense_batch_to_tilesparse(tensors: list[Tensor]) -> list[BitsparseTensor]:
    """Convert standalone tensors while synchronizing their value counts once."""
    if not tensors:
        return []
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all tensors in a compression batch must share a device")

    prepared = [_prepare_tiles(tensor) for tensor in tensors]
    counts = torch.stack([metadata.prefix[-1] for metadata in prepared])
    nnz_values = counts.cpu().tolist()
    return [
        _finish_tilesparse(tensor, metadata, None, int(nnz))
        for tensor, metadata, nnz in zip(tensors, prepared, nnz_values)
    ]


def FFN_backward(ctx, grad_output: Tensor):
    """Compute FFN gradients."""
    batch_shape = grad_output.shape[:-1]
    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    x, W1, W2 = ctx.saved_tensors
    h: BitsparseTensor = ctx.h_sparse
    ctx.h_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, h)

    grad_h = grad_output @ W2
    grad_z = mask_with_bitmask_(grad_h, h)
    del h

    if needs_x:
        grad_x = grad_z @ W1
        grad_x = grad_x.reshape(*batch_shape, -1)
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_backward_sparse(ctx, grad_output: Tensor):
    """Compute FFN gradients while keeping the hidden gradient tile-sparse."""
    batch_shape = grad_output.shape[:-1]
    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    x, W1, W2 = ctx.saved_tensors
    relu_sparse = ctx.h_sparse
    ctx.h_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, relu_sparse)
    grad_z = relu_layer_grad(grad_output, W2, relu_sparse)
    del relu_sparse

    grad_W1 = AspB_block(x.T, grad_z).T
    if needs_x:
        grad_x = spAB_block(grad_z, W1)
        grad_x = grad_x.reshape(*batch_shape, -1)
    else:
        grad_x = None
    return grad_x, grad_W1, grad_W2, None


def FFN3_backward(ctx, grad_output: Tensor):
    """Compute 3-layer ReLU FFN gradients using sparse saved activations."""
    x, W1, W2, W3 = ctx.saved_tensors
    h1 = ctx.h1_sparse
    z2 = ctx.h2_sparse
    ctx.h1_sparse = None
    ctx.h2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = AspB(grad_output.T, z2)

    grad_h2 = grad_output @ W3
    grad_z2 = mask_with_bitmask_(grad_h2, z2)
    del z2, grad_h2
    grad_W2 = AspB_block(grad_z2.T, h1)

    # grad_z1 = grad_z2 @ W2
    # Semi inplace variant to reduce peak memory.
    grad_h1 = inplace_mm_(grad_z2, W2)

    del grad_z2
    grad_z1 = mask_with_bitmask_(grad_h1, h1)
    del h1, grad_h1

    grad_x = grad_z1 @ W1 if needs_x else None
    grad_W1 = grad_z1.T @ x

    return grad_x, grad_W1, grad_W2, grad_W3, None


def FFN_relu2_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``.
        grad_output.shape = [*bs, in_dim]
    """
    bs_dims = grad_output.shape[:-1]          # [*bs, in_dim]
    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    x, W1, W2 = ctx.saved_tensors
    h = ctx.h_sparse
    ctx.h_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, h) # AspRelu2B_block(grad_output.T, z) #

    grad_h2 = grad_output @ W2
    grad_z = relu2_grad_sparse_(grad_h2, h)
    del h

    if needs_x:
        grad_x = grad_z @ W1
        grad_x = grad_x.reshape(*bs_dims, -1)
    else:
        grad_x = None
    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
    """ReLU² backward with the hidden gradient kept tile-sparse."""
    batch_shape = grad_output.shape[:-1]
    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    x, W1, W2 = ctx.saved_tensors
    relu_sparse = ctx.h_sparse
    ctx.h_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, relu_sparse)
    grad_z = relu2_layer_grad(grad_output, W2, relu_sparse)
    del relu_sparse

    grad_W1 = AspB_block(x.T, grad_z).T
    if needs_x:
        grad_x = spAB_block(grad_z, W1)
        grad_x = grad_x.reshape(*batch_shape, -1)
    else:
        grad_x = None

    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_3_backward(ctx, grad_output: Tensor):
    """Backward for ``z1 = k*relu(a1)^2``, ``z2 = k*relu(a2)^2`` using sparse caches."""
    x, W1, W2, W3 = ctx.saved_tensors
    h1 = ctx.h1_sparse
    h2 = ctx.h2_sparse
    ctx.h1_sparse = None
    ctx.h2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = AspRelu2B_block(grad_output.T, h2)

    grad_h2_sq = grad_output @ W3
    grad_z2 = relu2_grad_sparse_(grad_h2_sq, h2)
    del h2, grad_h2_sq

    grad_W2 = AspRelu2B_block(grad_z2.T, h1)

    grad_h1_sq = inplace_mm_(grad_z2, W2)        # grad_z1 = grad_z2 @ W2
    del grad_z2

    grad_z1 = relu2_grad_sparse_(grad_h1_sq, h1)
    del h1, grad_h1_sq

    grad_W1 = grad_z1.T @ x
    del x
    grad_x = grad_z1 @ W1 if needs_x else None
    del grad_z1

    return grad_x, grad_W1, grad_W2, grad_W3, None
