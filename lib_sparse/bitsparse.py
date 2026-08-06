import torch
from torch import Tensor

from .code.bitpacking import packed_nbytes

# Constant for RELU^2 scaling
RELU2_SCALE = 1
BLOCK_M = 64        # Rows per tile
BLOCK_N = 64        # Columns per tile


class BitsparseTensor:
    """Tile-wise bitmask sparse tensor for a dense matrix of shape ``shape``.

    ``vals`` stores positive entries in row-major tile order, ``bitmask`` marks
    nonzero locations with one bit per element, and ``prefix[t]`` gives the
    starting logical-value offset of tile ``t`` in ``vals``.

    ``vals_offset`` is an optional int64 tensor giving the starting logical-value offset
    of this layer's values inside a shared ``vals`` buffer.  When ``None`` (the
    default), a zero tensor is created so the tensor is self-contained.
    """
    vals: Tensor
    bitmask: Tensor
    prefix: Tensor
    vals_offset: Tensor
    BLOCK_M: int
    BLOCK_N: int
    grid_m: int
    grid_n: int

    def __init__(self, vals, bitmask, prefix,
                 grid_m, grid_n, BLOCK_M, BLOCK_N, shape,
                 vals_offset=None, packed_15bit=False, value_dtype=None):
        """Store compressed values and tile metadata for later unpack/masking."""
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        if vals_offset is None:
            vals_offset = torch.tensor(0, device=vals.device, dtype=torch.int64)
        self.vals_offset = vals_offset
        self.packed_15bit = packed_15bit
        self.value_dtype = vals.dtype if value_dtype is None else value_dtype
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, "
                f"nnz={self.nnz()}, sparsity={self.sparsity_ratio():.2f}, "
                f"packed_15bit={self.packed_15bit})")

    def nnz(self):
        """Count set bitmask bits. Intended for diagnostics, not hot paths."""
        bits = torch.arange(8, device=self.bitmask.device, dtype=torch.uint8)
        return ((self.bitmask[:, None] >> bits) & 1).sum()

    def vram_size(self):
        nnz = int(self.prefix[-1].item())
        val_size = packed_nbytes(nnz) if self.packed_15bit else nnz * self.vals.element_size()
        bitmask_size = self.bitmask.element_size() * self.bitmask.nelement()
        prefix_size = self.prefix.element_size() * self.prefix.nelement()
        return (val_size + bitmask_size + prefix_size) / 1024 ** 2

    def sparsity_ratio(self):
        return 1 - self.nnz() / (self.shape[0] * self.shape[1])


class TensorBuffer:
    vals: Tensor
    offset: Tensor

    def __init__(self, size: int, device="cuda", dtype=torch.bfloat16,
                 packed_15bit: bool = False):
        """ size: number of storage bytes in buffer
            device: device of buffer
            dtype: logical datatype of stored values"""
        self.size = size
        self.device = device
        self.dtype = dtype
        self.packed_15bit = packed_15bit

        # Init storage tensors
        if packed_15bit:
            storage_size = (self.size + 3) // 4 * 4 + 4
            self.vals = torch.zeros(storage_size, device=self.device, dtype=torch.uint8)
        else:
            if self.size % torch.empty((), dtype=dtype).element_size() != 0:
                raise ValueError("raw TensorBuffer size must be a multiple of dtype size")
            numel = self.size // torch.empty((), dtype=dtype).element_size()
            self.vals = torch.zeros(numel, device=self.device, dtype=self.dtype)
        self.offset = torch.zeros(1, device=self.device, dtype=torch.int64)

    def reset_buffer(self):
        """ Set offset tensor inside main training loop, since this needs to be consistent. """
        self.offset = torch.zeros(1, device=self.device, dtype=torch.int64)

    def to_state(self):
        """ Returns state in current buffer, decomposed into its objects for reloading. Useful for torch ops. """
        return self.size, self.device, self.dtype, self.packed_15bit, self.vals, self.offset

    @staticmethod
    def from_state(size, device, dtype, packed_15bit, vals, offset) -> TensorBuffer:
        """ Creates a TensorBuffer instance from its state. """
        buffer = TensorBuffer(size, device, dtype, packed_15bit)
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
