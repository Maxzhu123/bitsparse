import torch
from torch import Tensor

from src.code.bitpacking import packed_nbytes, packed_storage_nbytes

# Constant for RELU^2 scaling
RELU2_SCALE = 1
BLOCK_M = 64        # Rows per tile
BLOCK_N = 64        # Columns per tile


class _TileSparseTensor:
    """Metadata shared by packed activations and signed sparse gradients."""
    vals: Tensor
    bitmask: Tensor
    prefix: Tensor
    vals_offset: Tensor
    dtype: torch.dtype
    BLOCK_M: int
    BLOCK_N: int
    grid_m: int
    grid_n: int

    def __init__(self, vals, bitmask, prefix,
                 grid_m, grid_n, BLOCK_M, BLOCK_N, shape,
                 vals_offset, dtype):
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape
        self.dtype = dtype
        self.vals_offset = vals_offset

    def __repr__(self):
        return (f"{type(self).__name__}(shape={list(self.shape)}, "
                f"nnz={self.prefix[-1]}, sparsity={self.sparsity_ratio():.2f})")

    def vram_size(self):
        return self.nbytes() / 1024 ** 2

    def sparsity_ratio(self):
        return 1 - self.prefix[-1] / (self.shape[0] * self.shape[1])


class BitsparseTensor(_TileSparseTensor):
    """Tile-wise bitmask sparse tensor for a dense matrix of shape ``shape``.

    ``vals`` stores positive entries as a continuous stream of 15-bit values in
    row-major tile order, ``bitmask`` marks nonzero locations with one bit per
    element, and ``prefix[t]`` gives the logical value offset of tile ``t``.

    ``vals_offset`` is an int32 tensor giving the logical starting offset in
    ``vals``. It is zero for self-contained tensors and advances when ``vals``
    belongs to a shared :class:`TensorBuffer`.
    """
    def nbytes(self):
        """Return bytes attributable to this tensor's sparse payload and metadata."""
        nnz = int(self.prefix[-1].item())
        val_size = packed_nbytes(nnz)
        return val_size + self.bitmask.nbytes + self.prefix.nbytes + self.vals_offset.nbytes


class SparseGradientTensor(_TileSparseTensor):
    """Tile-sparse signed values produced by fused activation backpropagation.

    This uses the same bitmask and prefix layout as :class:`BitsparseTensor`,
    but stores ordinary 16-bit values. Gradients may be negative, so they
    cannot use the positive-only 15-bit representation without losing data.
    """
    def nbytes(self):
        nnz = int(self.prefix[-1].item())
        return (nnz * self.vals.element_size() + self.bitmask.nbytes
                + self.prefix.nbytes + self.vals_offset.nbytes)


TileSparseTensor = BitsparseTensor | SparseGradientTensor


class TensorBuffer:
    """Preallocated storage for packed values shared across sparse tensors."""

    vals: Tensor
    offset: Tensor

    def __init__(self, size: int, device="cuda", dtype=torch.bfloat16):
        """Allocate capacity for ``size`` logical 15-bit values."""
        self.size = size
        self.device = device
        self.dtype = dtype

        if self.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("TensorBuffer supports only bfloat16 and float16 values")
        self.vals = torch.zeros(
            packed_storage_nbytes(self.size), device=self.device, dtype=torch.uint8
        )
        self.offset = torch.zeros(1, device=self.device, dtype=torch.int32)

    def reset_buffer(self):
        """Clear packed storage and reset the next logical value offset."""
        self.vals.zero_()
        self.offset.zero_()

    def to_state(self):
        """Return the state needed to reconstruct this buffer."""
        return self.size, self.device, self.dtype, self.vals, self.offset

    @staticmethod
    def from_state(size, device, dtype, vals, offset) -> TensorBuffer:
        """Reconstruct a buffer from :meth:`to_state` output."""
        buffer = TensorBuffer(size, device, dtype)
        buffer.vals = vals
        buffer.offset = offset
        return buffer


def tile_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> tuple[int, int, int, int, int]:
    """Return tile-grid dimensions and tile storage sizes for a dense matrix shape."""
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n
    return grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES


@torch.no_grad()
def inplace_mm_(A, W, B=2048):
    """ A <- AW inplace operation. Done with batches. """
    m, n = A.shape

    x = torch.empty((B, n), device=A.device, dtype=A.dtype)
    y = torch.empty_like(x)

    for i in range(0, m, B):
        b = min(B, m - i)
        x[:b].copy_(A[i:i+b])
        torch.mm(x[:b], W, out=y[:b])
        A[i:i+b].copy_(y[:b])
    return A
