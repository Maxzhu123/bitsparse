"""Compare bitsparse encode/decode with the same format implemented in PyTorch.

BF16 input is 1 GiB at the default shape, with roughly 50% ReLU sparsity.
Occupancy masks are packed; values retain all 16 bits (no sign-bit packing).
Both methods allocate standalone outputs, without a shared TensorBuffer.
"""

import csv
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.benchmark import decompress
from lib_sparse.bitsparse import BitsparseTensor, tile_grid
from lib_sparse.config import BLOCK_M, BLOCK_N
from lib_sparse.src.functions import dense_to_tilesparse

BATCH_SIZE = 65_536
WIDTH = 8_192
DTYPE = torch.bfloat16
# Bound PyTorch indexing/scan workspaces without launching one operation per tile.
TILE_ROWS_PER_CHUNK = 128


def encode(data, sparse=False):
    if sparse:
        return dense_to_tilesparse(data, scale=None, pack_sbit=False)

    rows, columns = data.shape
    grid_m, grid_n, num_tiles, tile_numel, _ = tile_grid(rows, columns, BLOCK_M, BLOCK_N)
    if rows % BLOCK_M or columns % BLOCK_N:
        data = F.pad(data, (0, grid_n * BLOCK_N - columns, 0, grid_m * BLOCK_M - rows))
    tiles = data.reshape(grid_m, BLOCK_M, grid_n, BLOCK_N).permute(0, 2, 1, 3)
    mask = (tiles > 0).reshape(num_tiles, tile_numel)
    shifts = torch.arange(8, device=data.device, dtype=torch.uint8)
    bitmask = (mask.reshape(-1, 8).to(torch.uint8) << shifts).sum(dim=1, dtype=torch.uint8)
    # Count packed bytes instead of promoting the full boolean mask to int32.
    byte_counts = bitmask - ((bitmask >> 1) & 0x55)
    byte_counts = (byte_counts & 0x33) + ((byte_counts >> 2) & 0x33)
    byte_counts.add_(byte_counts >> 4).bitwise_and_(0x0F)
    counts = byte_counts.reshape(num_tiles, -1).sum(dim=1, dtype=torch.int32)
    del byte_counts
    prefix = torch.zeros(num_tiles + 1, device=data.device, dtype=torch.uint32)
    torch.cumsum(counts, dim=0, out=prefix.view(torch.int32)[1:])
    chunk_tiles = TILE_ROWS_PER_CHUNK * grid_n
    # Read allocation size and chunk offsets before compaction.
    offsets = prefix[::chunk_tiles].tolist()
    if num_tiles % chunk_tiles:
        offsets.append(prefix[-1].item())
    vals = torch.empty(offsets[-1], device=data.device, dtype=data.dtype)
    for chunk, first in enumerate(range(0, num_tiles, chunk_tiles)):
        last = min(first + chunk_tiles, num_tiles)
        chunk_mask = mask[first:last].flatten()
        chunk_vals = tiles[first // grid_n:(last + grid_n - 1) // grid_n].reshape(-1)
        # Known counts avoid the per-chunk synchronization of boolean indexing.
        indices = torch.nonzero_static(chunk_mask, size=offsets[chunk + 1] - offsets[chunk]).flatten()
        torch.index_select(chunk_vals, 0, indices, out=vals[offsets[chunk]:offsets[chunk + 1]])
        del chunk_vals, indices
    return BitsparseTensor(
        vals, bitmask, prefix, (rows, columns), data.dtype,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        vals_offset=torch.zeros((), device=data.device, dtype=torch.int64),
        pack_sbit=False,
    )


def decode(data, sparse=False):
    if sparse:
        return decompress(data)

    weights = 1 << torch.arange(8, device=data.vals.device, dtype=torch.uint8)
    dense = torch.empty((data.grid_m * data.BLOCK_M, data.grid_n * data.BLOCK_N),
                        device=data.vals.device, dtype=data.dtype)
    tile_numel = data.BLOCK_M * data.BLOCK_N
    chunk_tiles = TILE_ROWS_PER_CHUNK * data.grid_n
    num_tiles = data.grid_m * data.grid_n
    offsets = data.prefix[::chunk_tiles].tolist()
    if num_tiles % chunk_tiles:
        offsets.append(data.prefix[-1].item())
    for chunk, first in enumerate(range(0, num_tiles, chunk_tiles)):
        last = min(first + chunk_tiles, num_tiles)
        bits = data.bitmask[first * tile_numel // 8:last * tile_numel // 8]
        mask = (bits[:, None] & weights).bool().flatten()
        indices = torch.nonzero_static(mask, size=offsets[chunk + 1] - offsets[chunk]).flatten()
        tiles = torch.zeros(mask.numel(), device=data.vals.device, dtype=data.dtype)
        tiles.index_copy_(0, indices, data.vals[offsets[chunk]:offsets[chunk + 1]])
        del indices
        tiles = tiles.reshape(-1, data.grid_n, data.BLOCK_M, data.BLOCK_N)
        start_row = first // data.grid_n * data.BLOCK_M
        end_row = last // data.grid_n * data.BLOCK_M
        # Copy through a strided view instead of allocating another dense layout.
        dense[start_row:end_row].view(-1, data.BLOCK_M, data.grid_n, data.BLOCK_N).copy_(
            tiles.permute(0, 2, 1, 3)
        )
        del mask, tiles
    return dense[:data.shape[0], :data.shape[1]].contiguous()


@torch.no_grad()
def validate(data):
    baseline = encode(data)
    compressed = encode(data, sparse=True)
    for name in ("vals", "bitmask", "prefix", "vals_offset"):
        assert torch.equal(getattr(baseline, name), getattr(compressed, name)), name
    # Cross-decode: each decoder must also accept the other encoder's output.
    assert torch.equal(decode(compressed), data), "PyTorch decode differs"
    assert torch.equal(decode(baseline, sparse=True), data), "Bitsparse decode differs"


@torch.no_grad()
def run_step(data, operation, sparse=False, steps=1):
    """Measure time (ms) and peak allocated memory (MiB), including outputs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation(data, sparse=sparse)
        end.record()
        events.append((start, end))
        del output
    torch.cuda.synchronize()
    vram = torch.cuda.max_memory_allocated() / 2**20
    avg_time = sum(start.elapsed_time(end) for start, end in events) / steps
    return vram, avg_time


@torch.no_grad()
def evaluate(bs, operation, warmup_steps=5, eval_steps=32):
    if warmup_steps < 1 or eval_steps < 1:
        raise ValueError("Warmup and evaluation steps must be positive")
    generator = torch.Generator(device="cuda").manual_seed(0)
    data = torch.randn((bs, WIDTH), generator=generator, device="cuda", dtype=DTYPE).relu_()
    validate(data)
    if operation is decode:
        data = encode(data, sparse=True)

    run_step(data, operation, steps=warmup_steps)
    vram_base, avg_time_base = run_step(data, operation, steps=eval_steps)
    print(f"Baseline: peak={vram_base:.2f} MiB, avg_time={avg_time_base:.3f} ms")

    run_step(data, operation, sparse=True, steps=warmup_steps)
    vram, avg_time = run_step(data, operation, sparse=True, steps=eval_steps)
    print(f"Bitsparse: peak={vram:.2f} MiB, avg_time={avg_time:.3f} ms")
    return vram_base, avg_time_base, vram, avg_time


def run_batch(warmup_steps, eval_steps, batch_sizes=None,
              save_name="results/ablation_encode_decode.csv"):
    if batch_sizes is None:
        batch_sizes = [BATCH_SIZE]
    save_name = Path(save_name)
    save_name.parent.mkdir(parents=True, exist_ok=True)
    write_header = not save_name.exists() or save_name.stat().st_size == 0
    with save_name.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["operation", "method", "batch_size", "width", "vram", "avg_time"])
        for bs in batch_sizes:
            for operation in (encode, decode):
                print("-" * 50)
                print(f"{operation.__name__}: {bs = }, width={WIDTH}, dtype={DTYPE}", flush=True)
                vram_base, avg_time_base, vram, avg_time = evaluate(bs, operation, warmup_steps, eval_steps)
                writer.writerow([operation.__name__, "pytorch", bs, WIDTH, vram_base, avg_time_base])
                writer.writerow([operation.__name__, "bitsparse", bs, WIDTH, vram, avg_time])
                f.flush()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.cuda.set_device(0)
    # Check empty/full masks and partial tiles in addition to the measured input.
    validate(torch.zeros((BLOCK_M + 1, BLOCK_N + 1), device="cuda", dtype=DTYPE))
    validate(torch.ones((BLOCK_M + 1, BLOCK_N + 1), device="cuda", dtype=DTYPE))
    run_batch(warmup_steps=5, eval_steps=32,
              save_name=Path(__file__).resolve().parent / "results" / "ablation_encode_decode.csv")
