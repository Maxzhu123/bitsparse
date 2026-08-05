"""Utilities for packing non-negative 16-bit floating-point values into 15 bits.

The sparse activation format omits the zero sign bit from every stored BF16
value.  Logical value indices therefore address a continuous 15-bit stream,
not elements of the underlying uint8 storage tensor.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import triton
import triton.language as tl


BITS_PER_VALUE = 15


def packed_nbytes(numel: int) -> int:
    """Return the number of payload bytes needed for ``numel`` values."""
    return (int(numel) * BITS_PER_VALUE + 7) // 8


def packed_storage_nbytes(numel: int) -> int:
    """Return word-aligned storage with one guard word for vectorized reads."""
    payload = packed_nbytes(numel)
    return ((payload + 3) // 4) * 4 + 4


@triton.jit
def load_15bit_at_indices(
    packed_ptr,
    value_indices,
    mask,
    BF16: tl.constexpr,
):
    """Load logical values through aligned uint16 packed-stream windows.

    Splitting the index into groups avoids forming ``value_index * 15``,
    which can overflow int32 before the resulting storage index does.
    """
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


@triton.jit
def _compress_15bit_kernel(
    input_ptr,
    output_ptr,
    numel,
    compressed_numel,
    BLOCK_SIZE: tl.constexpr,
):
    output_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
    tl.store(output_ptr + output_offsets, packed & 0xFF, mask=output_mask)


@triton.jit
def _uncompress_15bit_kernel(
    input_ptr,
    output_ptr,
    numel,
    compressed_numel,
    BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < numel
    within_group = output_offsets % 8
    byte_indices = ((output_offsets // 8) * 15
                    + (within_group * 15) // 8)
    shifts = (within_group * 15) % 8
    byte0 = tl.load(
        input_ptr + byte_indices,
        mask=output_mask & (byte_indices < compressed_numel),
        other=0,
    ).to(tl.uint32)
    byte1 = tl.load(
        input_ptr + byte_indices + 1,
        mask=output_mask & ((byte_indices + 1) < compressed_numel),
        other=0,
    ).to(tl.uint32)
    byte2 = tl.load(
        input_ptr + byte_indices + 2,
        mask=output_mask & ((byte_indices + 2) < compressed_numel),
        other=0,
    ).to(tl.uint32)
    bits = ((byte0 | (byte1 << 8) | (byte2 << 16)) >> shifts) & 0x7FFF
    bits = bits.to(tl.uint16)
    if BF16:
        values = bits.to(tl.bfloat16, bitcast=True)
    else:
        values = bits.to(tl.float16, bitcast=True)
    tl.store(output_ptr + output_offsets, values, mask=output_mask)


def compress_15bit(data: Tensor) -> Tensor:
    """Pack non-negative CUDA BF16/FP16 values into an exact-size byte tensor."""
    if not data.is_cuda:
        raise ValueError("15-bit packing requires a CUDA tensor")
    if data.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("15-bit packing supports only bfloat16 and float16")
    if torch.any(data < 0).item():
        raise ValueError("15-bit packing can only omit the sign bit of non-negative values")

    contiguous = data.contiguous()
    numel = contiguous.numel()
    compressed_numel = packed_nbytes(numel)
    output = torch.empty(compressed_numel, dtype=torch.uint8, device=data.device)
    if numel:
        grid = (triton.cdiv(compressed_numel, 512),)
        _compress_15bit_kernel[grid](
            contiguous.view(torch.uint16).reshape(-1),
            output,
            numel,
            compressed_numel,
            BLOCK_SIZE=512,
            num_warps=1,
            num_stages=1,
        )
    return output


def pack_15bit_into(data: Tensor, output: Tensor) -> None:
    """Pack contiguous values into a preallocated uint8 tensor.

    This internal operator performs no validation or host synchronization. The
    caller must provide contiguous BF16/FP16 values with zero sign bits and an
    output containing at least :func:`packed_nbytes` payload bytes.
    """
    numel = data.numel()
    if numel == 0:
        return
    compressed_numel = packed_nbytes(numel)
    grid = (triton.cdiv(compressed_numel, 512),)
    _compress_15bit_kernel[grid](
        data.view(torch.uint16),
        output,
        numel,
        compressed_numel,
        BLOCK_SIZE=512,
        num_warps=1,
        num_stages=1,
    )


def uncompress_15bit(
    compressed_tensor: Tensor,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
) -> Tensor:
    """Restore values previously produced by :func:`compress_15bit`."""
    if not compressed_tensor.is_cuda:
        raise ValueError("15-bit unpacking requires a CUDA tensor")
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("15-bit unpacking supports only bfloat16 and float16")

    numel = math.prod(shape)
    expected_bytes = packed_nbytes(numel)
    if compressed_tensor.numel() < expected_bytes:
        raise ValueError(
            f"compressed tensor has {compressed_tensor.numel()} bytes; "
            f"expected at least {expected_bytes}"
        )

    output = torch.empty(numel, dtype=dtype, device=compressed_tensor.device)
    if numel:
        grid = (triton.cdiv(numel, 1024),)
        _uncompress_15bit_kernel[grid](
            compressed_tensor.contiguous().reshape(-1),
            output,
            numel,
            expected_bytes,
            BF16=dtype == torch.bfloat16,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=1,
        )
    return output.reshape(shape)
