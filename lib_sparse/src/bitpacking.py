"""Triton helpers for continuous bitstreams of non-negative values.

The packed codec drops the always-zero sign bit from each stored value (ReLU
activations are non-negative), then packs the remaining bits continuously:

  bf16 (15 bits/value)  — 16-bit bf16 with the sign bit dropped.
  fp8  (7 bits/value)   — 8-bit e4m3fn/e5m2 with the sign bit dropped.

The bitstream is addressed in uint16 words; ``packed_nbytes`` gives the byte
footprint of ``numel`` packed values.
"""

import triton
import triton.language as tl


_PACK_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
]


def packed_nbytes(numel: int, bits_per_value: int) -> int:
    """Return payload bytes for ``numel`` values of ``bits_per_value`` bits."""
    return (int(numel) * bits_per_value + 7) // 8


def packed_storage_nbytes(numel: int, bits_per_value: int) -> int:
    """Return word-aligned storage with one guard word for vectorized reads."""
    return ((packed_nbytes(numel, bits_per_value) + 3) // 4) * 4 + 4


@triton.jit
def load_packed_at_indices(packed_ptr, value_indices, mask, CODEC: tl.constexpr):
    """Decode logical indices from a continuous stream viewed as uint16 words.

    CODEC selects the value width and reinterpretation:
      0 = bf16 (15 bits, bitcast bf16), 1 = e4m3fn, 2 = e5m2 (7 bits each).
    """
    bits = 15 if CODEC == 0 else 7
    word_indices = (value_indices * bits) // 16
    shifts = (value_indices * bits) % 16
    word0 = tl.load(packed_ptr + word_indices, mask=mask, other=0).to(tl.uint32)
    word1 = tl.load(packed_ptr + word_indices + 1, mask=mask, other=0).to(tl.uint32)
    value = ((word0 | (word1 << 16)) >> shifts) & ((1 << bits) - 1)
    if CODEC == 0:
        return value.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    elif CODEC == 1:
        return value.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    else:
        return value.to(tl.uint8).to(tl.float8e5, bitcast=True)


@triton.autotune(configs=_PACK_CONFIGS, key=["num_tiles", "CODEC"], cache_results=True)
@triton.jit
def _pack_kernel(
    input_ptr,
    output_ptr,
    output_offset_ptr,
    tile_prefix_ptr,
    first_tile,
    num_tiles,
    CODEC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack a tile chunk into an arbitrary position in the bitstream."""
    bits = 15 if CODEC == 0 else 7
    prefix_start = tl.load(tile_prefix_ptr + first_tile).to(tl.int64)
    prefix_end = tl.load(tile_prefix_ptr + first_tile + num_tiles).to(tl.int64)
    numel = prefix_end - prefix_start

    logical_start = tl.load(output_offset_ptr).to(tl.int64) + prefix_start
    start_bit = logical_start * bits
    first_byte = start_bit // 8
    start_shift = (start_bit % 8).to(tl.int32)
    total_bits = numel * bits
    num_bytes = (start_shift + total_bits + 7) // 8

    byte_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = byte_offsets < num_bytes
    output_bytes = first_byte + byte_offsets

    # Every ``bits`` output bytes consume exactly eight values.  Computing the
    # source position within that repeating group keeps the lane-wise math in
    # 32 bits; only the absolute output address needs the 64-bit stream offset.
    bit_in_group = (byte_offsets % bits) * 8 - start_shift
    previous_value = bit_in_group < 0
    input_indices = (
        (byte_offsets // bits) * 8
        + tl.where(previous_value, -1, bit_in_group // bits)
    )
    bit_offsets = tl.where(
        previous_value, bit_in_group + bits, bit_in_group % bits
    )

    # The first byte can begin partway through a destination byte, but its
    # source still starts at input value zero.
    first = byte_offsets == 0
    input_indices = tl.where(first, 0, input_indices)
    bit_offsets = tl.where(first, 0, bit_offsets)

    value0 = tl.load(
        input_ptr + input_indices,
        mask=output_mask & (input_indices < numel),
        other=0,
    ).to(tl.uint32)
    value1_mask = output_mask & ((input_indices + 1) < numel)
    if CODEC == 0:
        # A BF16 byte only straddles values when fewer than eight bits remain.
        value1_mask &= bit_offsets > bits - 8
    value1 = tl.load(
        input_ptr + input_indices + 1,
        mask=value1_mask,
        other=0,
    ).to(tl.uint32)
    packed = (value0 >> bit_offsets) | (value1 << (bits - bit_offsets))
    packed = (packed << tl.where(first, start_shift, 0)) & 0xFF

    end_shift = ((start_shift + total_bits) % 8).to(tl.int32)
    valid_lo = tl.where(first, start_shift, 0)
    valid_hi = tl.where(
        (byte_offsets == num_bytes - 1) & (end_shift != 0), end_shift, 8
    )
    valid_bits = ((1 << valid_hi) - 1) & ~((1 << valid_lo) - 1)

    boundary = valid_bits != 0xFF
    old = tl.load(
        output_ptr + output_bytes,
        mask=output_mask & boundary,
        other=0,
    ).to(tl.uint32)
    merged = (old & ~valid_bits) | (packed & valid_bits)
    tl.store(output_ptr + output_bytes, merged, mask=output_mask)
