import time

import torch
from torch import Tensor

from lib_sparse.bitsparse import BitsparseTensor, TensorBuffer, bits_per_value
from lib_sparse.src.functions import dense_to_tilesparse
from lib_sparse.src.triton_operators import unpack_batch_


DEFAULT_SHAPES = ((1000, 4096), (15000, 4096), (21504, 8000))
PACK_SBIT = True
USE_BUFFER = True
BENCHMARK_DTYPE = torch.bfloat16


def generate_data(
    shape: tuple[int, int],
    generator: torch.Generator,
    dtype: torch.dtype,
    device: str,
    sparsity: float,
) -> Tensor:
    """Create a non-negative dense matrix with approximately ``sparsity`` zeros.

    Data is generated in BF16 (``randn`` has no FP8 kernel) and cast to the
    requested dtype, so FP8 inputs round-trip through storage losslessly.
    """
    data = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device).abs_()
    zero_mask = torch.rand(shape, generator=generator, device=device) < sparsity
    data.masked_fill_(zero_mask, 0)
    if dtype != torch.bfloat16:
        data = data.to(dtype)
    return data


def decompress(sparse: BitsparseTensor) -> Tensor:
    """Decompress all tiles in ``sparse`` into a newly allocated dense tensor."""
    rows, columns = sparse.shape
    output = torch.empty(
        (rows, columns), device=sparse.vals.device, dtype=sparse.dtype
    )
    unpack_batch_(
        sparse,
        output,
        first_m_tile=0,
        grid_n=sparse.grid_n,
        K=columns,
        batch_rows=rows,
        num_tiles_in_batch=sparse.grid_m * sparse.grid_n,
    )
    return output


def compress_batch(
    tensors: list[Tensor],
    pack_sbit: bool,
    buffer: TensorBuffer | None = None,
) -> list[BitsparseTensor]:
    """Compress a batch, reusing shared value storage when supplied."""
    if buffer is not None:
        buffer.reset_buffer()
    return [
        dense_to_tilesparse(
            tensor,
            scale=None,
            sparse_data=buffer,
            pack_sbit=pack_sbit,
        )
        for tensor in tensors
    ]


def decompress_batch(tensors: list[BitsparseTensor]) -> list[Tensor]:
    """Decompress every BitsparseTensor in a batch."""
    return [decompress(tensor) for tensor in tensors]


def benchmark_shape(
    tensors: list[Tensor],
    iters: int,
    warmup: int,
    pack_sbit: bool,
    use_buffer: bool = False,
) -> tuple[list[BitsparseTensor], list[Tensor], float, float, float]:
    """Time batched compression followed by batched decompression."""
    buffer = None
    if use_buffer:
        # Reserve worst-case NNZ capacity outside timing. Packed allocations
        # can each add up to seven values of offset alignment padding.
        capacity = sum(tensor.numel() for tensor in tensors)
        if pack_sbit:
            capacity += 7 * len(tensors)
            size = (capacity * bits_per_value(tensors[0].dtype) + 7) // 8
        else:
            size = capacity * tensors[0].element_size()
        buffer = TensorBuffer(
            size, device=tensors[0].device, dtype=tensors[0].dtype,
            pack_sbit=pack_sbit,
        )

    for _ in range(warmup):
        compressed = compress_batch(tensors, pack_sbit, buffer)
        decompressed = decompress_batch(compressed)
    torch.cuda.synchronize()

    compress_events = []
    decompress_events = []
    start_time = time.perf_counter()
    for _ in range(iters):
        compress_start = torch.cuda.Event(enable_timing=True)
        compress_end = torch.cuda.Event(enable_timing=True)
        decompress_end = torch.cuda.Event(enable_timing=True)

        compress_start.record()
        compressed = compress_batch(tensors, pack_sbit, buffer)
        compress_end.record()
        decompressed = decompress_batch(compressed)
        decompress_end.record()

        compress_events.append((compress_start, compress_end))
        decompress_events.append((compress_end, decompress_end))

    torch.cuda.synchronize()
    roundtrip_ms = (time.perf_counter() - start_time) * 1000
    compress_ms = sum(start.elapsed_time(end) for start, end in compress_events)
    decompress_ms = sum(start.elapsed_time(end) for start, end in decompress_events)
    return compressed, decompressed, compress_ms, decompress_ms, roundtrip_ms


def main() -> None:
    shapes = DEFAULT_SHAPES
    n = 8
    sp_ratios = [0.5, 0.8]
    iters = 32
    warmup = 5
    dtype = BENCHMARK_DTYPE
    device = "cuda"

    generator = torch.Generator(device=device).manual_seed(0)
    total_roundtrip_ms = 0.0
    print(f"  Dtype={dtype}, Pack_sbit={PACK_SBIT}, Use_buffer={USE_BUFFER}")

    for sparsity in sp_ratios:
        print(f"  Target_sparsity={sparsity:.1%}")
        for shape in shapes:
            tensors = [generate_data(shape, generator, dtype, device, sparsity) for _ in range(n)]

            compressed, decompressed, compress_ms, decompress_ms, roundtrip_ms = benchmark_shape(
                tensors, iters, warmup, PACK_SBIT, USE_BUFFER
            )

            # Correctness check
            for decompressed_tensor, original in zip(decompressed, tensors):
                # Compare bytes for an exact round-trip check that also works
                # for FP8, without requiring a floating-point equality kernel.
                assert decompressed_tensor.dtype == original.dtype
                assert torch.equal(
                    decompressed_tensor.view(torch.uint8), original.view(torch.uint8)
                )
            # Logical compressed footprint, excluding unused buffer capacity.
            storage_ratios = []
            for ct, original in zip(compressed, tensors):
                value_bytes = ct.vram_size()
                storage_ratio = value_bytes / (original.numel() * original.element_size())
                storage_ratios.append(storage_ratio)
            storage_ratio = sum(storage_ratios) / n

            compress_ms = compress_ms / iters / n
            decompress_ms = decompress_ms / iters / n
            roundtrip_ms = roundtrip_ms / iters / n

            print(
                f"shape={shape}, "
                f"compress={compress_ms:.3f} ms, decompress={decompress_ms:.3f} ms, "
                f"roundtrip={roundtrip_ms:.3f} ms, total_storage={storage_ratio:.4%}"
            )
            total_roundtrip_ms += roundtrip_ms

    print("Passed")
    print(f"Total time: {total_roundtrip_ms:.3f} ms")


if __name__ == "__main__":
    main()
