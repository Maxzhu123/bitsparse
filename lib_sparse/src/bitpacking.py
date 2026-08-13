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
    end_bit = start_bit + numel * bits
    first_byte = start_bit // 8
    end_byte = (end_bit + 7) // 8

    byte_offsets = (tl.program_id(0).to(tl.int64) * BLOCK_SIZE
                    + tl.arange(0, BLOCK_SIZE).to(tl.int64))
    output_mask = byte_offsets < (end_byte - first_byte)
    output_bytes = first_byte + byte_offsets

    relative_bit = output_bytes * 8 - start_bit
    source_bit = tl.maximum(relative_bit, 0)
    input_indices = source_bit // bits
    bit_offsets = source_bit % bits

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
    packed = (value0 >> bit_offsets) | (value1 << (bits - bit_offsets))
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
