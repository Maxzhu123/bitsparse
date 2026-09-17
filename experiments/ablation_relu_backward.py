"""Compare ReLU and ReLU² backward: PyTorch vs bitsparse fused kernels.
"""

import csv
import gc
from pathlib import Path

import torch
from lib_sparse.bitsparse import tile_grid
from lib_sparse.config import BLOCK_M, BLOCK_N, RELU2_SCALE
from lib_sparse.src.functions import dense_to_tilesparse
from lib_sparse.src.triton_kernels import _relu_grad_sparse_kernel
from lib_sparse.src.triton_operators import tile_pack, relu2_grad_sparse_

BATCH_SIZE = 16_000
WIDTH = 21_504
DTYPE = torch.bfloat16


def make_bitmask(activation):
    """Use the production tile packer; discard counts after mask preparation."""
    rows, columns = activation.shape
    grid_m, grid_n, num_tiles, tile_numel, tile_bytes = tile_grid(
        rows, columns, BLOCK_M, BLOCK_N
    )
    counts = torch.empty(num_tiles, device=activation.device, dtype=torch.int32)
    bitmask = torch.empty(num_tiles * tile_bytes, device=activation.device, dtype=torch.uint8)
    tile_pack(activation, counts, bitmask, rows, columns, grid_m, grid_n,
              BLOCK_M, BLOCK_N, tile_numel, tile_bytes)
    return bitmask


def backward(grad, saved, sparse=False, relu2=False):
    if relu2:
        if sparse:
            return relu2_grad_sparse_(grad, saved)
        return grad * (2.0 * RELU2_SCALE * saved)
    if not sparse:
        return torch.ops.aten.threshold_backward.grad_input(
            grad, saved, 0, grad_input=grad
        )
    rows, columns = grad.shape
    grid_m, grid_n, _, _, tile_bytes = tile_grid(rows, columns, BLOCK_M, BLOCK_N)
    # Same launch as mask_with_bitmask_, without retaining unused sparse values.
    _relu_grad_sparse_kernel[(grid_m, grid_n)](
        grad, saved, rows, columns,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, TILE_BYTES=tile_bytes,
    )
    return grad


@torch.no_grad()
def validate(shape, relu2=False):
    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.randn(shape, generator=generator, device="cuda", dtype=DTYPE).relu_()
    grad = torch.randn(shape, generator=generator, device="cuda", dtype=DTYPE)
    saved = dense_to_tilesparse(activation, scale=None) if relu2 else make_bitmask(activation)
    expected = backward(grad.clone(), activation, relu2=relu2)
    actual = backward(grad, saved, sparse=True, relu2=relu2)
    assert torch.equal(actual, expected), "Backward outputs differ"


@torch.no_grad()
def run_step(grad, saved, sparse=False, steps=1, relu2=False):
    """Measure backward time (ms) and peak allocated memory (MiB)."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = []
    for _ in range(steps):
        # ReLU² is not idempotent. Reset outside timing without a backup tensor.
        if relu2:
            grad.fill_(1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        backward(grad, saved, sparse, relu2)
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    vram = torch.cuda.max_memory_allocated() / 2**20
    avg_time = sum(start.elapsed_time(end) for start, end in events) / steps
    return vram, avg_time


@torch.no_grad()
def evaluate(bs, warmup_steps=5, eval_steps=32, relu2=False):
    """Prepare inputs, check correctness, then compare dense and sparse backward."""
    if warmup_steps < 1 or eval_steps < 1:
        raise ValueError("Warmup and evaluation steps must be positive")
    shape = (bs, WIDTH)
    # Include a partial tile case as well as the measured shape.
    validate((BLOCK_M + 1, BLOCK_N + 1), relu2)
    validate(shape, relu2)

    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.randn(shape, generator=generator, device="cuda", dtype=DTYPE).relu_()
    grad = torch.randn(shape, generator=generator, device="cuda", dtype=DTYPE)

    # Dense baseline
    run_step(grad, activation, steps=warmup_steps, relu2=relu2)
    vram_base, avg_time_base = run_step(grad, activation, steps=eval_steps, relu2=relu2)
    print(f"Baseline: peak={vram_base:.2f} MiB, avg_time={avg_time_base:.3f} ms")

    # ReLU needs only the mask; ReLU² also reads compact BF16 values and prefix.
    saved = dense_to_tilesparse(activation, scale=None) if relu2 else make_bitmask(activation)
    del activation
    run_step(grad, saved, sparse=True, steps=warmup_steps, relu2=relu2)
    vram, avg_time = run_step(grad, saved, sparse=True, steps=eval_steps, relu2=relu2)
    print(f"Bitsparse: peak={vram:.2f} MiB, avg_time={avg_time:.3f} ms")
    return vram_base, avg_time_base, vram, avg_time


def run_batch(warmup_steps, eval_steps, batch_sizes=None, save_name="results/ablation_relu_backward.csv",
              relu2=False):
    if batch_sizes is None:
        batch_sizes = [BATCH_SIZE]
    save_name = Path(save_name)
    save_name.parent.mkdir(parents=True, exist_ok=True)
    write_header = not save_name.exists() or save_name.stat().st_size == 0
    with save_name.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["method", "batch_size", "width", "vram", "avg_time"])
        for bs in batch_sizes:
            print("-" * 50)
            print(f"{bs = }, width={WIDTH}, dtype={DTYPE}, relu2={relu2}")
            vram_base, avg_time_base, vram, avg_time = evaluate(bs, warmup_steps, eval_steps, relu2)
            writer.writerow(["pytorch", bs, WIDTH, vram_base, avg_time_base])
            writer.writerow(["bitsparse", bs, WIDTH, vram, avg_time])
            f.flush()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.cuda.set_device(0)
    run_batch(warmup_steps=5, eval_steps=32,
              save_name=Path(__file__).resolve().parent / "results" / "ablation_relu_backward.csv")
    run_batch(warmup_steps=5, eval_steps=32, relu2=True,
              save_name=Path(__file__).resolve().parent / "results" / "ablation_relu2_backward.csv")
