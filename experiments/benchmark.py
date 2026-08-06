import time

import torch
from torch import Tensor

from bitsparse import BitsparseTensor
from lib_sparse.code.functions import dense_to_tilesparse
from lib_sparse.code.bitpacking import packed_nbytes
from lib_sparse.code.triton_operators import unpack_batch_


DEFAULT_SHAPES = ((1000, 4096), (4000, 4096), (15000, 4096))
PACKED_15BIT = True


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
        (rows, columns), device=sparse.vals.device, dtype=sparse.value_dtype
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


def compress_batch(tensors: list[Tensor], packed_15bit: bool) -> list[BitsparseTensor]:
    """Compress every dense tensor in a batch."""
    return [
        dense_to_tilesparse(tensor, packed_15bit=packed_15bit)
        for tensor in tensors
    ]


def decompress_batch(tensors: list[BitsparseTensor]) -> list[Tensor]:
    """Decompress every BitsparseTensor in a batch."""
    return [decompress(tensor) for tensor in tensors]


def benchmark_shape(
    tensors: list[Tensor], iters: int, warmup: int, packed_15bit: bool
) -> tuple[list[BitsparseTensor], list[Tensor], float, float, float]:
    """Time batched compression followed by batched decompression."""
    for _ in range(warmup):
        compressed = compress_batch(tensors, packed_15bit)
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
        compressed = compress_batch(tensors, packed_15bit)
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
    dtype = torch.bfloat16
    device = "cuda"

    generator = torch.Generator(device=device).manual_seed(0)
    total_roundtrip_ms = 0.0
    print(f"  Packed_15bit={PACKED_15BIT}")

    for sparsity in sp_ratios:
        print(f"  Target_sparsity={sparsity:.1%}")
        for shape in shapes:
            tensors = [generate_data(shape, generator, dtype, device, sparsity) for _ in range(n)]

            compressed, decompressed, compress_ms, decompress_ms, roundtrip_ms = benchmark_shape(
                tensors, iters, warmup, PACKED_15BIT
            )

            # Correctness check
            for decompressed_tensor, original in zip(decompressed, tensors):
                torch.testing.assert_close(decompressed_tensor, original, rtol=0, atol=0)
            # Compression check
            storage_ratios = []
            for ct, original in zip(compressed, tensors):
                nnz = ct.prefix[-1].item()
                value_bytes = packed_nbytes(nnz) if ct.packed_15bit else nnz * original.element_size()
                storage_ratio = value_bytes / (original.numel() * original.element_size())
                storage_ratios.append(storage_ratio)
            storage_ratio = sum(storage_ratios) / n

            compress_ms = compress_ms / iters / n
            decompress_ms = decompress_ms / iters / n
            roundtrip_ms = roundtrip_ms / iters / n

            print(
                f"shape={shape}, "
                f"compress={compress_ms:.3f} ms, decompress={decompress_ms:.3f} ms, "
                f"roundtrip={roundtrip_ms:.3f} ms, value_storage={storage_ratio:.1%}"
            )
            total_roundtrip_ms += roundtrip_ms

    print("Passed")
    print(f"Total time: {total_roundtrip_ms:.3f} ms")


if __name__ == "__main__":
    main()
