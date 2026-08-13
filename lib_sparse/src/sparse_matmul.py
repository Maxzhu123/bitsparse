import torch
from torch import Tensor

from ..fp8 import matmul
from .triton_operators import unpack_batch_, unpack_relu2_batch_
from ..bitsparse import BitsparseTensor


def AspB(A: Tensor, B_sparse: BitsparseTensor, A_scale: Tensor|None=None, row_batch: int = 0) -> Tensor:
    """Compute ``A @ B`` where ``B`` is stored as ``BitsparseTensor``.

    Shapes: ``A[M, N]`` and sparse ``B[N, K]`` produce ``out[M, K]``.
    Unpacks ``B`` to dense before matmul.

    By default the whole sparse ``B`` is unpacked at once.  When ``row_batch > 0``,
    rows of ``B`` are unpacked in tile-aligned batches of at most ``row_batch`` rows
    and accumulated, which trades a bit of overhead for lower peak VRAM.
    """
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_m, grid_n = B_sparse.grid_m, B_sparse.grid_n
    N, K = B_sparse.shape
    M = A.shape[0]

    if row_batch <= 0:
        num_tiles = grid_m * grid_n
        dense = torch.empty(N, K, device=A.device, dtype=B_sparse.dtype)
        unpack_batch_(B_sparse, dense, 0, grid_n, K, N, num_tiles)
        return matmul(A, dense, B_sparse.fp8, a_scale=A_scale, b_scale=B_sparse.scale)

    # Blockwise path: tile-aligned row batches so tiles are never split.
    out = torch.zeros(M, K, device=A.device, dtype=A.dtype)

    row_tiles_per_batch = max(1, row_batch // BLOCK_M)
    for first_n_tile in range(0, grid_m, row_tiles_per_batch):
        n_start = first_n_tile * BLOCK_M
        n_end = min(n_start + row_tiles_per_batch * BLOCK_M, N)
        batch_rows = n_end - n_start
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, K, device=A.device, dtype=B_sparse.dtype)
        unpack_batch_(B_sparse, dense_batch, first_n_tile, grid_n, K, batch_rows,
                      num_tiles_in_batch)
        A_batch = A[:, n_start:n_end]
        out.addmm_(A_batch, dense_batch)

    return out


def spAB(A_sparse: BitsparseTensor, B: Tensor, out: Tensor | None = None, row_batch: int = 0) -> Tensor:
    """Compute ``A_sparse @ B`` by unpacking the sparse matrix once.

    Shapes: sparse ``A[M, N]`` and ``B[N, K]`` produce ``out[M, K]``.

    By default the whole sparse ``A`` is unpacked at once and the result is
    returned.  When ``row_batch > 0``, rows of ``A`` are unpacked in tile-aligned
    batches of at most ``row_batch`` rows and accumulated into ``out`` (which
    must be provided and shaped ``[M, K]``), trading a bit of overhead for lower
    peak VRAM.
    """
    vals = A_sparse.vals
    BLOCK_M, BLOCK_N = A_sparse.BLOCK_M, A_sparse.BLOCK_N
    grid_m, grid_n = A_sparse.grid_m, A_sparse.grid_n
    M, N = A_sparse.shape
    K = B.shape[1]

    if row_batch <= 0:
        num_tiles = grid_m * grid_n
        dense = torch.empty(M, N, device=B.device, dtype=A_sparse.dtype)
        unpack_batch_(A_sparse, dense, 0, grid_n, N, M, num_tiles)
        return matmul(dense, B, A_sparse.fp8, a_scale=A_sparse.scale)

    # Blockwise path: tile-aligned row batches so tiles are never split.
    out = torch.zeros(M, K, device=B.device, dtype=B.dtype) if out is None else out

    row_tiles_per_batch = max(1, row_batch // BLOCK_M)
    for first_m_tile in range(0, grid_m, row_tiles_per_batch):
        m_start = first_m_tile * BLOCK_M
        m_end = min(m_start + row_tiles_per_batch * BLOCK_M, M)
        batch_rows = m_end - m_start
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=B.device, dtype=A_sparse.dtype)
        unpack_batch_(A_sparse, dense_batch, first_m_tile, grid_n, N, batch_rows,
                      num_tiles_in_batch)
        torch.mm(dense_batch, B, out=out[m_start:m_end])
    return out


def AspRelu2B(A: Tensor, B_sparse: BitsparseTensor, A_scale: Tensor|None=None, row_batch: int = 0) -> Tensor:
    """Compute A @ (k * B^2) where sparse B = relu(preact), elementwise square for activation.

    Shapes: ``A[M, N]`` and sparse ``B[N, K]`` produce ``out[M, K]``.
    Unpacks ``k * B^2`` to dense before matmul.  The squared reconstruction is
    materialised in BF16 (squaring overflows the FP8 storage range) and then
    re-quantised by the FP8 matmul.

    By default the whole sparse ``B`` is unpacked at once.  When ``row_batch > 0``,
    rows of ``B`` are unpacked in tile-aligned batches of at most ``row_batch`` rows
    and accumulated, which trades a bit of overhead for lower peak VRAM.
    """
    vals = B_sparse.vals
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_m, grid_n = B_sparse.grid_m, B_sparse.grid_n
    N, K = B_sparse.shape
    M = A.shape[0]

    if row_batch <= 0:
        num_tiles = grid_m * grid_n
        dense = torch.empty(N, K, device=A.device, dtype=torch.bfloat16)
        unpack_relu2_batch_(B_sparse, dense, 0, grid_n, K, N, num_tiles)
        return matmul(A, dense, B_sparse.fp8, a_scale=A_scale)

    # Blockwise path: tile-aligned row batches so tiles are never split.
    out = torch.zeros(M, K, device=A.device, dtype=A.dtype)

    row_tiles_per_batch = max(1, row_batch // BLOCK_M)
    for first_n_tile in range(0, grid_m, row_tiles_per_batch):
        n_start = first_n_tile * BLOCK_M
        n_end = min(n_start + row_tiles_per_batch * BLOCK_M, N)
        batch_rows = n_end - n_start
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, K, device=A.device, dtype=torch.bfloat16)
        unpack_relu2_batch_(B_sparse, dense_batch, first_n_tile, grid_n, K, batch_rows,
                            num_tiles_in_batch)
        A_batch = A[:, n_start:n_end]
        out.addmm_(A_batch, dense_batch)

    return out
