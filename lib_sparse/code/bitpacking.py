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
def load_15bit_at_indices(packed_ptr, value_indices, mask, BF16: tl.constexpr):
    """Decode logical indices from a continuous stream viewed as uint16 words."""
    within_group = value_indices % 16
    word_indices = ((value_indices // 16) * 15
                    + (within_group * 15) // 16)
    shifts = (within_group * 15) % 16
    word0 = tl.load(packed_ptr + word_indices, mask=mask, other=0).to(tl.uint32)
    word1 = tl.load(packed_ptr + word_indices + 1, mask=mask, other=0).to(tl.uint32)
    bits = ((word0 | (word1 << 16)) >> shifts) & 0x7FFF
    bits = bits.to(tl.uint16)
    if BF16:
        values = bits.to(tl.bfloat16, bitcast=True)
    else:
        values = bits.to(tl.float16, bitcast=True)
    return values


@triton.autotune(configs=_PACK_15BIT_CONFIGS, key=["numel_index"], cache_results=True)
@triton.jit
def _pack_15bit_kernel(
    input_ptr,
    output_ptr,
    output_offset_ptr,
    numel_ptr,
    numel_index,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack contiguous 16-bit values at a byte-aligned logical buffer offset."""
    numel = tl.load(numel_ptr + numel_index).to(tl.uint32)
    compressed_numel = (numel // 8) * 15 + ((numel % 8) * 15 + 7) // 8
    output_offsets = (tl.program_id(0).to(tl.uint32) * BLOCK_SIZE
                      + tl.arange(0, BLOCK_SIZE).to(tl.uint32))
    output_mask = output_offsets < compressed_numel
    within_group = output_offsets % 15
    input_indices = ((output_offsets // 15) * 8
                     + (within_group * 8) // 15)
    bit_offsets = (within_group * 8) % 15

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

    # Packed activations start at multiples of eight logical values: eight
    # 15-bit values occupy exactly fifteen bytes. Standalone storage supplies
    # a zero offset through the same path.
    logical_offset = tl.load(output_offset_ptr)
    output_base = (logical_offset // 8) * 15
    tl.store(output_ptr + output_base + output_offsets, packed & 0xFF,
             mask=output_mask)
