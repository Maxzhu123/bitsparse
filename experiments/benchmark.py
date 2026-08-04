"""Benchmark BitsparseTensor compression and decompression.

Run from the repository root with ``PYTHONPATH=lib_sparse`` set, for example::

    PYTHONPATH=lib_sparse python experiments/benchmark.py
"""

import time

import torch
from torch import Tensor

from src.bitsparse import BitsparseTensor
from src.code.functions import dense_to_tilesparse
from src.code.triton_operators import unpack_batch_


DEFAULT_SHAPES = ((1024, 4096), (4096, 4096), (16384, 4096))


def generate_data(
    shape: tuple[int, int],
    generator: torch.Generator,
    dtype: torch.dtype,
    device: str,
    sparsity: float,
) -> Tensor:
    """Create a non-negative dense matrix with approximately ``sparsity`` zeros."""
    data = torch.randn(shape, generator=generator, dtype=dtype, device=device).abs_()
    zero_mask = torch.rand(shape, generator=generator, device=device) < sparsity
    data.masked_fill_(zero_mask, 0)
    return data


def decompress(sparse: BitsparseTensor) -> Tensor:
    """Decompress all tiles in ``sparse`` into a newly allocated dense tensor."""
    rows, columns = sparse.shape
    output = torch.empty(
        (rows, columns), device=sparse.vals.device, dtype=sparse.vals.dtype
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


def run_batch(
    tensors: list[Tensor],
) -> tuple[list[BitsparseTensor], list[Tensor]]:
    """Compress the whole batch first, then decompress the whole batch."""
    compressed = [dense_to_tilesparse(tensor) for tensor in tensors]
    decompressed = [decompress(sparse) for sparse in compressed]
    return compressed, decompressed


def benchmark_shape(
    tensors: list[Tensor], iters: int, warmup: int
) -> tuple[list[BitsparseTensor], list[Tensor], float, float, float]:
    """Time batched compression followed by batched decompression."""
    for _ in range(warmup):
        compressed, decompressed = run_batch(tensors)
    torch.cuda.synchronize()

    compress_events = []
    decompress_events = []
    start_time = time.perf_counter()
    for _ in range(iters):
        compress_start = torch.cuda.Event(enable_timing=True)
        compress_end = torch.cuda.Event(enable_timing=True)
        decompress_end = torch.cuda.Event(enable_timing=True)

        compress_start.record()
        compressed = [dense_to_tilesparse(tensor) for tensor in tensors]
        compress_end.record()
        decompressed = [decompress(sparse) for sparse in compressed]
        decompress_end.record()

        compress_events.append((compress_start, compress_end))
        decompress_events.append((compress_end, decompress_end))

    torch.cuda.synchronize()
    roundtrip_ms = (time.perf_counter() - start_time) * 1000 / iters
    compress_ms = sum(start.elapsed_time(end) for start, end in compress_events) / iters
    decompress_ms = sum(start.elapsed_time(end) for start, end in decompress_events) / iters
    return compressed, decompressed, compress_ms, decompress_ms, roundtrip_ms


def main() -> None:
    shapes = DEFAULT_SHAPES
    n = 16
    sparsity = 0.5
    iters = 100
    warmup = 5
    dtype = torch.bfloat16
    device = "cuda"

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA because Bitsparse uses Triton kernels.")

    generator = torch.Generator(device=device).manual_seed(0)

    print(
        f"device={torch.cuda.get_device_name()}, dtype={dtype}, "
        f"batch_size={n}, target_sparsity={sparsity:.1%}, iters={iters}"
    )
    total_roundtrip_ms = 0.0
    for shape in shapes:
        tensors = [
            generate_data(shape, generator, dtype, device, sparsity)
            for _ in range(n)
        ]
        compressed, decompressed, compress_ms, decompress_ms, roundtrip_ms = benchmark_shape(
            tensors, iters, warmup
        )

        for decompressed_tensor, original in zip(decompressed, tensors):
            torch.testing.assert_close(decompressed_tensor, original, rtol=0, atol=0)

        total_roundtrip_ms += roundtrip_ms
        nnz = sum(torch.count_nonzero(tensor).item() for tensor in tensors)
        dense_mib = sum(tensor.nbytes for tensor in tensors) / 1024**2
        compressed_mib = sum(sparse.vram_size() for sparse in compressed)
        print(
            f"shape={shape},"
            f"compress={compress_ms:.3f} ms, decompress={decompress_ms:.3f} ms, "
            f"roundtrip={roundtrip_ms:.3f} ms"
        )

    print("Passed")
    print(f"Total time: {total_roundtrip_ms:.3f} ms")


if __name__ == "__main__":
    main()
