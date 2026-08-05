import torch
from torch import Tensor

from src.code.triton_kernels import (
    _tile_pack_kernel,
    _compact_vals_kernel,
    _compact_vals_15bit_kernel,
    _unpack_batch_kernel,
    _unpack_values_batch_kernel,
    _unpack_relu2_batch_kernel,
    _relu_grad_sparse_kernel,
    _relu2_grad_sparse_kernel,
    _relu2_layer_grad_kernel,
    _relu_layer_grad_kernel,
)
from src.bitsparse import (
    RELU2_SCALE,
    BitsparseTensor,
    SparseGradientTensor,
    TileSparseTensor,
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
    dense: Tensor, tile_prefix: Tensor, vals: Tensor, vals_offset: Tensor,
    M: int, N: int, grid_n: int, num_tiles: int,
    BLOCK_M: int, BLOCK_N: int, TILE_NUMEL: int,
) -> None:
    """Scatter positive values into a contiguous 16-bit staging tensor."""
    _compact_vals_kernel[(num_tiles,)](
        dense, tile_prefix, vals, vals_offset,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=8, num_stages=1,
    )


def compact_vals_15bit(
    dense: Tensor, tile_prefix: Tensor, vals: Tensor, vals_offset: Tensor,
    M: int, N: int, grid_n: int, num_tiles: int,
    BLOCK_M: int, BLOCK_N: int, TILE_NUMEL: int,
) -> None:
    """Compact directly into a shared zero-initialized 15-bit stream."""
    _compact_vals_15bit_kernel[(num_tiles,)](
        dense, tile_prefix, vals.view(torch.int32), vals_offset,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=8, num_stages=2,
    )


def unpack_batch_(
    sparse: TileSparseTensor, output: Tensor,
    first_m_tile: int, grid_n: int, K: int, batch_rows: int,
    num_tiles_in_batch: int,
) -> Tensor:
    """Unpack slice of sparse tiles into a dense output``batch_rows x K`` slice (in-place)."""
    BLOCK_M = sparse.BLOCK_M
    BLOCK_N = sparse.BLOCK_N
    if isinstance(sparse, BitsparseTensor):
        _unpack_batch_kernel[(num_tiles_in_batch,)](
            sparse.vals.view(torch.uint16), sparse.bitmask, sparse.prefix,
            sparse.vals_offset, output,
            first_m_tile, grid_n, K, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=BLOCK_M * BLOCK_N,
            TILE_BYTES=BLOCK_M * BLOCK_N // 8,
            BF16=sparse.dtype == torch.bfloat16,
            num_warps=4, num_stages=1,
        )
    else:
        _unpack_values_batch_kernel[(num_tiles_in_batch,)](
            sparse.vals, sparse.bitmask, sparse.prefix, sparse.vals_offset, output,
            first_m_tile, grid_n, K, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=BLOCK_M * BLOCK_N,
            TILE_BYTES=BLOCK_M * BLOCK_N // 8,
            num_warps=4, num_stages=1,
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
    _unpack_relu2_batch_kernel[(num_tiles_in_batch,)](
        sparse.vals.view(torch.uint16), sparse.bitmask, sparse.prefix, sparse.vals_offset,
        output,
        first_m_tile, grid_n, K, batch_rows,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        BF16=sparse.dtype == torch.bfloat16,
        num_warps=4, num_stages=1,
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
        num_warps=4, num_stages=2,
    )
    return grad


def relu2_grad_sparse_(grad: Tensor, sparse_z: BitsparseTensor) -> Tensor:
    """Apply the ReLU² derivative in-place on ``grad`` using sparse ``z``.

    Computes `dpreact = grad * 2 * k * r` for active entries, where
    `z = k * r²` and `r = relu(a)` is stored sparsely.  `grad` is
    overwritten with the result and returned.
    """
    BLOCK_M = sparse_z.BLOCK_M
    BLOCK_N = sparse_z.BLOCK_N
    _relu2_grad_sparse_kernel[(sparse_z.grid_m, sparse_z.grid_n)](
        grad, sparse_z.vals.view(torch.uint16), sparse_z.bitmask,
        sparse_z.prefix, sparse_z.vals_offset,
        sparse_z.shape[0], sparse_z.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N,
        TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        BF16=sparse_z.dtype == torch.bfloat16,
        num_warps=8, num_stages=2,
    )
    return grad


def relu2_layer_grad(
    grad_output: Tensor,
    W2: Tensor,
    relu_sparse: BitsparseTensor,
) -> SparseGradientTensor:
    """Return sparse ``(grad_output @ W2) * d(relu²)/dz`` values.

    The output is an ordinary 16-bit sparse stream because these gradients are
    signed. The activation remains intact in its positive-only packed stream.
    """
    M, N = relu_sparse.shape
    nnz = int(relu_sparse.prefix[-1].item())
    values = torch.empty(nnz, device=grad_output.device, dtype=relu_sparse.dtype)
    values_offset = torch.zeros(1, device=grad_output.device, dtype=torch.int32)
    BLOCK_M = relu_sparse.BLOCK_M
    BLOCK_N = relu_sparse.BLOCK_N

    _relu2_layer_grad_kernel[(relu_sparse.grid_m, relu_sparse.grid_n)](
        grad_output, W2,
        relu_sparse.vals.view(torch.uint16), relu_sparse.bitmask,
        relu_sparse.prefix, relu_sparse.vals_offset, values,
        M, N, relu_sparse.grid_n,
        D=grad_output.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N,
        TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        BF16=relu_sparse.dtype == torch.bfloat16,
    )

    return SparseGradientTensor(
        values, relu_sparse.bitmask, relu_sparse.prefix,
        relu_sparse.grid_m, relu_sparse.grid_n,
        BLOCK_M, BLOCK_N, relu_sparse.shape,
        vals_offset=values_offset, dtype=relu_sparse.dtype,
    )


def relu_layer_grad(
    grad_output: Tensor,
    W2: Tensor,
    relu_sparse: BitsparseTensor,
) -> SparseGradientTensor:
    """Return sparse ``grad_output @ W2`` values at active ReLU positions."""
    M, N = relu_sparse.shape
    nnz = int(relu_sparse.prefix[-1].item())
    values = torch.empty(nnz, device=grad_output.device, dtype=relu_sparse.dtype)
    values_offset = torch.zeros(1, device=grad_output.device, dtype=torch.int32)
    BLOCK_M = relu_sparse.BLOCK_M
    BLOCK_N = relu_sparse.BLOCK_N

    _relu_layer_grad_kernel[(relu_sparse.grid_m, relu_sparse.grid_n)](
        grad_output, W2,
        relu_sparse.bitmask, relu_sparse.prefix, values,
        M, N, relu_sparse.grid_n,
        D=grad_output.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N,
        TILE_BYTES=BLOCK_M * BLOCK_N // 8,
    )

    return SparseGradientTensor(
        values, relu_sparse.bitmask, relu_sparse.prefix,
        relu_sparse.grid_m, relu_sparse.grid_n,
        BLOCK_M, BLOCK_N, relu_sparse.shape,
        vals_offset=values_offset, dtype=relu_sparse.dtype,
    )
