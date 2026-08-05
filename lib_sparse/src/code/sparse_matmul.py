import torch
from torch import Tensor

from src.code.triton_operators import unpack_batch_, unpack_relu2_batch_
from src.bitsparse import BitsparseTensor, TileSparseTensor

def AspB(A: Tensor, B_sparse: TileSparseTensor) -> Tensor:
    """Compute ``A @ B`` where ``B`` is stored tile-sparsely.

    Shapes: ``A[P, M]`` and sparse ``B[M, N]`` produce ``out[P, N]``.
    Unpacks ``B`` to dense before matmul.
    """
    grid_m, grid_n = B_sparse.grid_m, B_sparse.grid_n
    M, N = B_sparse.shape

    num_tiles = grid_m * grid_n
    dense = torch.empty(M, N, device=A.device, dtype=B_sparse.dtype)
    unpack_batch_(B_sparse, dense, 0, grid_n, N, M, num_tiles)
    return A @ dense


def AspB_block(A: Tensor, B_sparse: TileSparseTensor, row_batch: int = 2048) -> Tensor:
    """Compute ``A @ B_sparse`` blockwise to reduce peak VRAM.

    ``row_batch`` is the target number of rows per batch; it is rounded
    down to a tile-aligned boundary so tiles are never split across batches.

    Shapes: ``A[K, M]`` and sparse ``B[M, N]`` produce ``out[K, N]``.
    """
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_n = B_sparse.grid_n
    M, N = B_sparse.shape
    K = A.shape[0]

    out = torch.zeros(K, N, device=A.device, dtype=A.dtype)

    row_tiles_per_batch = max(1, row_batch // BLOCK_M)
    for first_m_tile in range(0, B_sparse.grid_m, row_tiles_per_batch):
        m_start = first_m_tile * BLOCK_M
        m_end = min(m_start + row_tiles_per_batch * BLOCK_M, M)
        batch_rows = m_end - m_start
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=A.device, dtype=B_sparse.dtype)
        unpack_batch_(B_sparse, dense_batch, first_m_tile, grid_n, N, batch_rows,
                      num_tiles_in_batch)
        A_batch = A[:, m_start:m_end]
        out.addmm_(A_batch, dense_batch)

    return out


def spAB(A_sparse: TileSparseTensor, B: Tensor) -> Tensor:
    """Compute ``A_sparse @ B`` by unpacking the sparse matrix once.

    Shapes: sparse ``A[M, N]`` and ``B[N, K]`` produce ``out[M, K]``.
    """
    grid_n = A_sparse.grid_n
    M, N = A_sparse.shape

    num_tiles = A_sparse.grid_m * grid_n
    dense = torch.empty(M, N, device=B.device, dtype=A_sparse.dtype)
    unpack_batch_(A_sparse, dense, 0, grid_n, N, M, num_tiles)
    return dense @ B


def spAB_block(A_sparse: TileSparseTensor, B: Tensor, row_batch: int = 2048, out: Tensor=None) -> Tensor:
    """Compute ``A_sparse @ B`` blockwise to reduce peak VRAM.

    Unpacks row batches of sparse ``A`` to dense, then multiplies by ``B``.

    Shapes: sparse ``A[M, N]`` and ``B[N, K]`` produce ``out[M, K]``.
    """
    BLOCK_M, BLOCK_N = A_sparse.BLOCK_M, A_sparse.BLOCK_N
    grid_n = A_sparse.grid_n
    M, N = A_sparse.shape
    K = B.shape[1]

    if out is None:
        out = torch.empty(M, K, device=B.device, dtype=B.dtype)

    for m_start in range(0, M, row_batch):
        m_end = min(m_start + row_batch, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=B.device, dtype=A_sparse.dtype)
        unpack_batch_(A_sparse, dense_batch, first_m_tile, grid_n, N, batch_rows,
                      num_tiles_in_batch)
        torch.mm(dense_batch, B, out=out[m_start:m_end])
    return out


def AspRelu2B(A: Tensor, B_sparse: BitsparseTensor) -> Tensor:
    """Compute A @ (k * B^2) where sparse B = relu(preact), elementwise square for activation.

    Shapes: ``A[P, M]`` and sparse ``B[M, N]`` produce ``out[P, N]``.
    Unpacks ``k * B^2`` to dense before matmul.
    """
    grid_m, grid_n = B_sparse.grid_m, B_sparse.grid_n
    M, N = B_sparse.shape

    num_tiles = grid_m * grid_n
    dense = torch.empty(M, N, device=A.device, dtype=B_sparse.dtype)
    unpack_relu2_batch_(B_sparse, dense, 0, grid_n, N, M, num_tiles)
    return A @ dense


def AspRelu2B_block(A: Tensor, B_sparse: BitsparseTensor, row_batch: int = 512) -> Tensor:
    """Compute A @ (k * B^2) by unpacking ReLU2 tiles in row batches.

    Shapes: ``A[K, M]`` and sparse ``B[M, N]`` produce ``out[K, N]``.
    """
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_n = B_sparse.grid_n
    M, N = B_sparse.shape
    K = A.shape[0]

    out = torch.zeros(K, N, device=A.device, dtype=A.dtype)

    for m_start in range(0, M, row_batch):
        m_end = min(m_start + row_batch, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=A.device, dtype=B_sparse.dtype)
        unpack_relu2_batch_(B_sparse, dense_batch, first_m_tile, grid_n, N, batch_rows,
                            num_tiles_in_batch)
        out.addmm_(A[:, m_start:m_end], dense_batch)

    return out
