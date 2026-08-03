from torch import Tensor

from src.code.triton_kernels import (
    _tile_pack_kernel,
    _compact_vals_kernel,
    _unpack_batch_kernel,
    _unpack_relu2_batch_kernel,
    _relu_grad_sparse_kernel,
    _relu2_grad_sparse_kernel,
    _relu2_layer_grad_kernel,
    _relu_layer_sparse_kernel,
)
from src.bitsparse import RELU2_SCALE, BitsparseTensor


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
    dense: Tensor, tile_prefix: Tensor, vals: Tensor, vals_offset: Tensor,
    M: int, N: int, grid_n: int, num_tiles: int,
    BLOCK_M: int, BLOCK_N: int, TILE_NUMEL: int,
) -> None:
    """Scatter positive dense values into compact ``vals`` buffer (in-place)."""
    _compact_vals_kernel[(num_tiles,)](
        dense, tile_prefix, vals, vals_offset,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
    )


def unpack_batch_(
    sparse: BitsparseTensor, output: Tensor,
    first_m_tile: int, grid_n: int, K: int, batch_rows: int,
    num_tiles_in_batch: int,
) -> Tensor:
    """Unpack slice of sparse tiles into a dense output``batch_rows x K`` slice (in-place)."""
    BLOCK_M = sparse.BLOCK_M
    BLOCK_N = sparse.BLOCK_N
    _unpack_batch_kernel[(num_tiles_in_batch,)](
        sparse.vals, sparse.bitmask, sparse.prefix, sparse.vals_offset,
        output,
        first_m_tile, grid_n, K, batch_rows,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        num_warps=8, num_stages=2,
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
        sparse.vals, sparse.bitmask, sparse.prefix, sparse.vals_offset,
        output,
        first_m_tile, grid_n, K, batch_rows,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
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
        grad, sparse_z.vals, sparse_z.bitmask, sparse_z.prefix, sparse_z.vals_offset,
        sparse_z.shape[0], sparse_z.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N,
        TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )
    return grad


def relu2_layer_grad(
    grad_output: Tensor, W2: Tensor, r_sparse: BitsparseTensor,
) -> None:
    """ Relu^2 backward layer, including linear and activation.
    Overwrite r_sparse values with ``dpreact = (grad_output @ W2) * 2*k*r``.
    BLOCK_K (inner-loop tile) is chosen by kernel autotune.
    """
    M, N = r_sparse.shape
    BLOCK_M = r_sparse.BLOCK_M
    BLOCK_N = r_sparse.BLOCK_N
    _relu2_layer_grad_kernel[(r_sparse.grid_m, r_sparse.grid_n)](
        grad_output, W2,
        r_sparse.vals, r_sparse.bitmask, r_sparse.prefix, r_sparse.vals_offset,
        r_sparse.vals,
        M, N, r_sparse.grid_n,
        D=grad_output.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
    )


def relu_layer_sparse_(
    grad_output: Tensor, W2: Tensor, z_sparse: BitsparseTensor,
) -> BitsparseTensor:
    """ Relu backward layer, including linear and activation.
        Combine grad_z = grad_output @ W2,
                grad_z = grad_z * (z>0)
                grad_z = sparse(grad_z)
        Overwrite z_sparse values.
        BLOCK_K (inner-loop tile) is chosen by kernel autotune.
    """
    M, N = z_sparse.shape
    BLOCK_M = z_sparse.BLOCK_M
    BLOCK_N = z_sparse.BLOCK_N
    _relu_layer_sparse_kernel[(z_sparse.grid_m, z_sparse.grid_n)](
        grad_output, W2,
        z_sparse.bitmask, z_sparse.prefix, z_sparse.vals_offset,
        z_sparse.vals,
        M, N, z_sparse.grid_n,
        D=grad_output.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=BLOCK_M * BLOCK_N, TILE_BYTES=BLOCK_M * BLOCK_N // 8,
    )

    return z_sparse