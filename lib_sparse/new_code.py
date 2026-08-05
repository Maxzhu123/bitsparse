import math

import torch
from torch import Tensor
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def _compress_15bit_kernel(
    input_ptr,          # uint16 input
    output_ptr,         # uint8 output
    numel,              # number of input values
    compressed_numel,   # number of output bytes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Treat the input as a continuous stream of 15-bit little-endian values and
    emit that stream as bytes.
    """
    output_offsets = (
        tl.program_id(axis=0) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    output_mask = output_offsets < compressed_numel

    # First bit represented by each output byte.
    bit_positions = output_offsets * 8
    input_indices = bit_positions // 15
    bit_offsets = bit_positions % 15

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

    # The requested byte may cross a 15-bit value boundary.
    packed = (value0 >> bit_offsets) | (
        value1 << (15 - bit_offsets)
    )

    tl.store(
        output_ptr + output_offsets,
        packed & 0xFF,
        mask=output_mask,
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def _uncompress_15bit_kernel(
    input_ptr,          # uint16 input (compressed bytes viewed as uint16)
    output_ptr,         # uint16 output
    numel,              # number of output values
    compressed_numel,   # number of input bytes
    BLOCK_SIZE: tl.constexpr,
):
    # Process in reverse so the first blocks read the tail of the byte stream,
    # which compress_fn wrote most recently and is still resident in L2.
    output_offsets = (
        numel - 1
        - (tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    )
    output_mask = output_offsets >= 0

    # k = 15*offs lands at bit k; window starts at bit 16*(k // 16).
    k = output_offsets * 15
    idx16 = k // 16
    shift = k % 16

    u0 = tl.load(
        input_ptr + idx16,
        mask=output_mask,
        other=0,
    ).to(tl.uint32)

    u1 = tl.load(
        input_ptr + idx16 + 1,
        mask=output_mask,
        other=0,
    ).to(tl.uint32)

    window = u0 | (u1 << 16)
    restored = (window >> shift) & 0x7FFF

    tl.store(
        output_ptr + output_offsets,
        restored,
        mask=output_mask,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_fn(data: Tensor, dtype=None, device=None) -> Tensor:
    """Remove and bit-pack the sign bit from positive fp16/bf16 data."""
    numel = data.numel()
    compressed_numel = (numel * 15 + 7) // 8

    bits = data.contiguous().view(torch.uint16).reshape(-1)

    # Pad so the byte stream can always be viewed as uint16 by the
    # uncompress kernel (which reads two uint16 per value).
    padded_numel = compressed_numel + 8
    if padded_numel & 1:
        padded_numel += 1
    output = torch.empty(
        padded_numel,
        dtype=torch.uint8,
        device=device,
    )

    grid = lambda meta: (triton.cdiv(compressed_numel, meta['BLOCK_SIZE']),)
    _compress_15bit_kernel[grid](bits, output, numel, compressed_numel)
    return output


def uncompress_fn(
    compressed_tensor: Tensor,
    shape: torch.Size,
    dtype,
    device=None,
) -> Tensor:
    """Restore fp16/bf16 data previously produced by ``compress_fn``."""
    numel = math.prod(shape)
    expected_bytes = (numel * 15 + 7) // 8

    restored_bits = torch.empty(
        numel,
        dtype=torch.uint16,
        device=device,
    )

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    _uncompress_15bit_kernel[grid](
        compressed_tensor.contiguous().reshape(-1).view(torch.uint16),
        restored_bits,
        numel,
        expected_bytes,
    )

    # Reinterpret the restored bits without performing a numeric conversion.
    return restored_bits.view(dtype).reshape(shape)
