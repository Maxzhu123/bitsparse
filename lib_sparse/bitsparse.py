import torch
from torch import Tensor

from .fp8 import is_fp8
from .src.bitpacking import packed_nbytes


def pack_codec(dtype) -> int:
    """Packed-bitstream codec for a storage dtype.

    0 = bf16 (15 bits/value), 1 = e4m3fn, 2 = e5m2 (7 bits/value).  The sign
    bit is dropped because stored values are non-negative ReLU activations.
    """
    if dtype == getattr(torch, "float8_e4m3fn", None):
        return 1
    if dtype == getattr(torch, "float8_e5m2", None):
        return 2
    return 0


def bits_per_value(dtype) -> int:
    """Number of bits each value occupies in the packed bitstream."""
    return 7 if pack_codec(dtype) else 15


class BitsparseTensor:
    """Tile-wise bitmask sparse tensor for a dense matrix of shape ``shape``.
    """
    vals: Tensor            # Positive entries in row-major tile order. dtype is the storage dtype.
    bitmask: Tensor         # One bit per element, in row-major tile order. uint8 packed dtype.
    prefix: Tensor          # Starting logical-value offset of each tile in ``vals``. uint32 dtype.
    vals_offset: Tensor     # Offset where values in this tensor start in vals. 0 if not using shared buffer
    dtype: torch.dtype    # dtype of the original (and reconstructed) tensor
    BLOCK_M: int
    BLOCK_N: int
    grid_m: int
    grid_n: int

    def __init__(self, vals, bitmask, prefix, shape, dtype,
                 grid_m, grid_n, BLOCK_M, BLOCK_N,
                 scale=None, vals_offset=None, pack_sbit=False):
        """Store compressed values and tile metadata for later unpack/masking."""
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        if vals_offset is None:
            vals_offset = torch.tensor(0, device=vals.device, dtype=torch.int64)
        self.vals_offset = vals_offset
        self.pack_sbit = pack_sbit
        self.dtype = dtype
        # Logical dtype of the stored values; for pack_sbit ``vals`` is a raw
        # uint8 bitstream, so the logical dtype must be carried separately.
        # self.input_dtype = vals.dtype if storage_dtype is None else storage_dtype
        self.scale = scale
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape

    @property
    def fp8(self) -> bool:
        """True when values are stored as FP8 (e4m3fn or e5m2) instead of BF16."""
        return is_fp8(self.dtype)

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, "
                f"nnz={self.nnz()}, sparsity={self.sparsity_ratio():.2f}, "
                f"pack_sbit={self.pack_sbit})")

    def nnz(self):
        """Count set bitmask bits. Intended for diagnostics, not hot paths."""
        bits = torch.arange(8, device=self.bitmask.device, dtype=torch.uint8)
        return ((self.bitmask[:, None] >> bits) & 1).sum()

    def vram_size(self):
        nnz = int(self.prefix[-1].item())
        if self.pack_sbit:
            val_size = packed_nbytes(nnz, bits_per_value(self.dtype))
        else:
            val_size = nnz * self.vals.element_size()
        bitmask_size = self.bitmask.element_size() * self.bitmask.nelement()
        prefix_size = self.prefix.element_size() * self.prefix.nelement()
        return val_size + bitmask_size + prefix_size

    def sparsity_ratio(self):
        return 1 - self.nnz() / (self.shape[0] * self.shape[1])


class TensorBuffer:
    vals: Tensor        # Value tensor
    offset: Tensor      # Next offset for where next element can go, tracking where the buffer is filled to.

    def __init__(self, size: int, device="cuda", dtype=torch.bfloat16,
                 pack_sbit: bool = False):
        """ size: number of storage bytes in buffer
            device: device of buffer
            dtype: logical datatype of stored values"""
        self.size = size
        self.device = device
        self.dtype = dtype
        self.pack_sbit = pack_sbit

        # Init storage tensors
        if pack_sbit:
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


def tile_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> tuple[int, int, int, int, int]:
    """Return tile-grid dimensions and tile storage sizes for a dense matrix shape."""
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n
    return grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES
