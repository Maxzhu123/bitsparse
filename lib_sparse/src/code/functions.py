import torch
from torch import Tensor

from src.code.triton_operators import (
    compact_vals,
    mask_with_bitmask_,
    relu2_grad_sparse_,
    relu2_layer_grad,
    tile_pack, relu_layer_sparse_
)
from src.code.sparse_matmul import AspB, AspB_block, AspRelu2B_block, AspRelu2B, spAB_block
from src.bitsparse import BitsparseTensor, TensorBuffer, inplace_mm_, tile_grid, BLOCK_M, BLOCK_N


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


def FFN_backward(ctx, grad_output: Tensor):
    """Compute FFN gradients."""
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
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_backward_sparse(ctx, grad_output: Tensor):
    """Compute FFN gradients while keeping grad_z in the existing bit-sparse storage."""
    x, W1, W2 = ctx.saved_tensors
    h = ctx.h_sparse
    ctx.h_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, h)
    # Combine grad_output @ W2, relu + masking. Updates h inplace.
    grad_z = relu_layer_sparse_(grad_output, W2, h)

    grad_W1 = AspB_block(x.T, grad_z).T
    if needs_x:
        grad_x = spAB_block(grad_z, W1)
    else:
        grad_x = None
    return grad_x, grad_W1, grad_W2, None


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


