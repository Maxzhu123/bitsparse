import torch
from torch import Tensor

# Constant for RELU^2 scaling
RELU2_SCALE = 1
BLOCK_M = 64        # Rows per tile
BLOCK_N = 64        # Columns per tile


class BitsparseTensor:
    """Tile-wise bitmask sparse tensor for a dense matrix of shape ``shape``.

    ``vals`` stores positive entries in row-major tile order, ``bitmask`` marks
    nonzero locations with one bit per element, and ``prefix[t]`` gives the
    starting offset of tile ``t`` in ``vals``.

    ``vals_offset`` is an optional int32 tensor giving the starting offset of
    this layer's values inside a shared ``vals`` buffer.  When ``None`` (the
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
                 vals_offset=None):
        """Store compressed values and tile metadata for later unpack/masking."""
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        if vals_offset is None:
            vals_offset = torch.tensor(0, device=vals.device, dtype=torch.int32)
        self.vals_offset = vals_offset
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, "
                f"nnz={self.prefix[-1]}, sparsity={self.sparsity_ratio():.2f})")

    def vram_size(self):
        val_size = self.vals.element_size() * self.prefix[-1]
        bitmask_size = self.bitmask.element_size() * self.bitmask.nelement()
        prefix_size = self.prefix.element_size() * self.prefix.nelement()
        return (val_size + bitmask_size + prefix_size) / 1024 ** 2

    def sparsity_ratio(self):
        return 1 - self.prefix[-1] / (self.shape[0] * self.shape[1])


class TensorBuffer:
    vals: Tensor
    offset: Tensor

    def __init__(self, size: int, device="cuda", dtype=torch.bfloat16):
        """ size: number of elements in buffer
            device: device of buffer
            dtype: datatype of buffer"""
        self.size = size
        self.device = device
        self.dtype = dtype

        # Init storage tensors
        self.vals = torch.zeros(self.size, device=self.device, dtype=self.dtype)
        self.offset = torch.zeros(1, device=self.device, dtype=torch.int32)

    def reset_buffer(self):
        """ Set offset tensor inside main training loop, since this needs to be consistent. """
        self.offset = torch.zeros(1, device=self.device, dtype=torch.int32)

    def to_state(self):
        """ Returns state in current buffer, decomposed into its objects for reloading. Useful for torch ops. """
        return self.size, self.device, self.dtype, self.vals, self.offset

    @staticmethod
    def from_state(size, device, dtype, vals, offset) -> TensorBuffer:
        """ Creates a TensorBuffer instance from its state. """
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

