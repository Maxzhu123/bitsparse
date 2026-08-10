"""Triton helpers for continuous streams of non-negative 15-bit values."""

import triton
import triton.language as tl


BITS_PER_VALUE = 15


_PACK_15BIT_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
]


def packed_nbytes(numel: int) -> int:
    """Return payload bytes for ``numel`` 15-bit values."""
    return (int(numel) * BITS_PER_VALUE + 7) // 8


def packed_storage_nbytes(numel: int) -> int:
    """Return word-aligned storage with one guard word for vectorized reads."""
    return ((packed_nbytes(numel) + 3) // 4) * 4 + 4


@triton.jit
def load_15bit_at_indices(packed_ptr, value_indices, mask):
    """Decode logical indices from a continuous stream viewed as uint16 words."""
    within_group = value_indices % 16
    word_indices = ((value_indices // 16) * 15
                    + (within_group * 15) // 16)
    shifts = (within_group * 15) % 16
    word0 = tl.load(packed_ptr + word_indices, mask=mask, other=0).to(tl.uint32)
    word1 = tl.load(packed_ptr + word_indices + 1, mask=mask, other=0).to(tl.uint32)
    bits = ((word0 | (word1 << 16)) >> shifts) & 0x7FFF
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.autotune(configs=_PACK_15BIT_CONFIGS, key=["num_tiles"], cache_results=True)
@triton.jit
def _pack_15bit_kernel(
    input_ptr,
    output_ptr,
    output_offset_ptr,
    tile_prefix_ptr,
    first_tile,
    num_tiles,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack a tile chunk into an arbitrary position in the 15-bit stream."""
    prefix_start = tl.load(tile_prefix_ptr + first_tile).to(tl.int64)
    prefix_end = tl.load(tile_prefix_ptr + first_tile + num_tiles).to(tl.int64)
    numel = prefix_end - prefix_start

    logical_start = tl.load(output_offset_ptr).to(tl.int64) + prefix_start
    start_bit = logical_start * 15
    end_bit = start_bit + numel * 15
    first_byte = start_bit // 8
    end_byte = (end_bit + 7) // 8

    byte_offsets = (tl.program_id(0).to(tl.int64) * BLOCK_SIZE
                    + tl.arange(0, BLOCK_SIZE).to(tl.int64))
    output_mask = byte_offsets < (end_byte - first_byte)
    output_bytes = first_byte + byte_offsets

    relative_bit = output_bytes * 8 - start_bit
    source_bit = tl.maximum(relative_bit, 0)
    input_indices = source_bit // 15
    bit_offsets = source_bit % 15

    value0 = tl.load(
        input_ptr + input_indices,
        mask=output_mask & (input_indices < numel),
        other=0,
    ).to(tl.uint32)
    value1 = tl.load(
        input_ptr + input_indices + 1,
        mask=output_mask & ((input_indices + 1) < numel),
        other=0,
    ).to(tl.uint32)
    packed = (value0 >> bit_offsets) | (value1 << (15 - bit_offsets))
    packed = (packed << tl.maximum(-relative_bit, 0)) & 0xFF

    byte_start_bit = output_bytes * 8
    valid_lo = tl.minimum(tl.maximum(start_bit - byte_start_bit, 0), 8)
    valid_hi = tl.minimum(tl.maximum(end_bit - byte_start_bit, 0), 8)
    valid_bits = ((1 << valid_hi) - 1) & ~((1 << valid_lo) - 1)

    boundary = valid_bits != 0xFF
    old = tl.load(
        output_ptr + output_bytes,
        mask=output_mask & boundary,
        other=0,
    ).to(tl.uint32)
    merged = (old & ~valid_bits) | (packed & valid_bits)
    tl.store(output_ptr + output_bytes, merged, mask=output_mask)
