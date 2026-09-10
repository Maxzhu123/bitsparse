"""
sparsity_analysis.py

Measure the counting sparsity (fraction of zero activations) of the ReLU²
feed-forward activations in a Nemotron-H checkpoint, over several sequences
drawn from the FineWeb10B validation shards.

Nemotron-H is a hybrid model: only the `layers_block_type == "mlp"` blocks
contain a `NemotronHMLP`, and its `act_fn` is `ReLUSquaredActivation`
(`relu(x)²`). We hook that activation module directly, so the measured tensor is
exactly what the FFN consumes.

For every MLP layer we track:
  * average sparsity - the running mean nonzero fraction across batches.
  * worst sparsity   - the densest (most nonzero) batch, i.e. the worst case
                       for a compression scheme that must fit every batch.

The tracker mirrors the counting-sparsity tracker used in
bitsparse/nanogpt/sparsity_analysis.py (itself ported from
optimizer/modded-nanogpt/try_gpt.py).
"""

import csv
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from llm import NemotronHForCausalLM


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
RESULTS_PATH = Path(__file__).resolve().parent / "sparsity.csv"

# FineWeb10B shards. Each file is a 256-int32 header followed by uint16 tokens.
DATA_ROOT = Path(
    os.environ.get("FINEWEB_ROOT", "/home/bubbles/Documents/bitsparse/nanogpt/data")
)
DATA_PATTERN = "fineweb10B/fineweb_val_*.bin"
SHARD_HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1

# The Mamba/RMSNorm hub kernels are only a speed optimisation and are not
# required for measuring sparsity, so fall back to the torch path when the
# `kernels` package is unavailable or older than 0.9.0.
USE_KERNELS = False

# Measurement config: each batch contains BATCH_SIZE sequences of SEQ_LEN tokens.
BATCH_SIZE = 1
SEQ_LEN = 4096
NUM_BATCHES = 64
# Warmup batches are excluded from NUM_BATCHES and the reported statistics.
WARMUP_STEPS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActivationSparsityTracker:
    """Accumulate the average and worst-case nonzero fraction of activations."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.average: dict[int, Tensor] = {}
        self.worst: dict[int, Tensor] = {}
        self.counts: dict[int, int] = {}
        self.batch_sparsity: dict[int, Tensor] = {}
        self.worst_batch_avg: Tensor | None = None

    @torch.no_grad()
    def update(self, layer_num: int, x: Tensor) -> None:
        ratio = ((x != 0).sum() / x.numel()).detach()

        if layer_num not in self.average:
            self.average[layer_num] = ratio
            self.worst[layer_num] = ratio
            self.counts[layer_num] = 1
        else:
            count = self.counts[layer_num]
            self.average[layer_num] += (ratio - self.average[layer_num]) / (count + 1)
            self.worst[layer_num] = torch.maximum(self.worst[layer_num], ratio)
            self.counts[layer_num] = count + 1

        self._update_batch(layer_num, ratio)

    def _update_batch(self, layer_num: int, ratio: Tensor) -> None:
        """Track the densest whole-model batch once every layer has reported."""
        self.batch_sparsity[layer_num] = ratio
        if len(self.batch_sparsity) != self.num_layers:
            return

        batch_avg = torch.stack(list(self.batch_sparsity.values())).mean()
        if self.worst_batch_avg is None:
            self.worst_batch_avg = batch_avg
        else:
            self.worst_batch_avg = torch.maximum(self.worst_batch_avg, batch_avg)
        self.batch_sparsity.clear()

    def reset(self) -> None:
        self.average.clear()
        self.worst.clear()
        self.counts.clear()
        self.batch_sparsity.clear()
        self.worst_batch_avg = None

    def average_as_floats(self) -> dict[int, float]:
        return {layer_num: value.item() for layer_num, value in sorted(self.average.items())}

    def worst_as_floats(self) -> dict[int, float]:
        return {layer_num: value.item() for layer_num, value in sorted(self.worst.items())}

    def average_overall(self) -> float:
        values = self.average_as_floats()
        return sum(values.values()) / max(len(values), 1)

    def worst_overall(self) -> float:
        values = self.worst_as_floats()
        return max(values.values(), default=float("nan"))

    def worst_batch_avg_as_float(self) -> float:
        if self.worst_batch_avg is None:
            return float("nan")
        return self.worst_batch_avg.item()


def mlp_layer_indices(model: NemotronHForCausalLM) -> list[int]:
    """Layer indices whose block type is an MLP (i.e. contains a ReLU² FFN)."""
    return [
        layer_idx
        for layer_idx, block in enumerate(model.model.layers)
        if block.block_type == "mlp"
    ]


def _make_hook(tracker: ActivationSparsityTracker, layer_num: int) -> Any:
    """Report the output of a block's ReLU² activation module."""

    def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        tracker.update(layer_num, output.detach())

    return hook


def attach_sparsity_tracker(
    model: NemotronHForCausalLM,
) -> tuple[ActivationSparsityTracker, list[Any]]:
    """Hook every MLP block's ReLU² activation and return the tracker + handles."""
    layer_indices = mlp_layer_indices(model)
    if not layer_indices:
        raise ValueError("Model contains no 'mlp' blocks to measure")

    tracker = ActivationSparsityTracker(len(layer_indices))
    handles = [
        model.model.layers[layer_idx].mixer.act_fn.register_forward_hook(
            _make_hook(tracker, layer_idx)
        )
        for layer_idx in layer_indices
    ]
    return tracker, handles


def load_shard(path: Path) -> Tensor:
    """Load one FineWeb10B .bin shard into CPU memory as uint16 tokens."""
    header = torch.from_file(str(path), False, SHARD_HEADER_INTS, dtype=torch.int32)
    if int(header[0]) != SHARD_MAGIC:
        raise ValueError(f"bad magic number in {path}")
    if int(header[1]) != SHARD_VERSION:
        raise ValueError(f"unsupported version in {path}")

    num_tokens = int(header[2])
    tokens = torch.empty(num_tokens, dtype=torch.uint16)
    with path.open("rb", buffering=0) as file:
        file.seek(SHARD_HEADER_INTS * 4)
        num_bytes = file.readinto(tokens.numpy())
    if num_bytes != 2 * num_tokens:
        raise ValueError(f"short read in {path}")
    return tokens


def sequence_batches(
    seq_len: int,
    batch_size: int,
    num_batches: int,
) -> Iterator[Tensor]:
    """Yield contiguous `batch_size` x `seq_len` windows from the shards."""
    files = sorted(DATA_ROOT.glob(DATA_PATTERN))
    if not files:
        raise FileNotFoundError(f"No shards matched {DATA_PATTERN!r} under {DATA_ROOT}")

    batch_tokens = batch_size * seq_len
    file_index = 0
    tokens = load_shard(files[file_index])
    pos = 0

    for _ in range(num_batches):
        # Advance across shard boundaries, wrapping around at the end.
        while pos + batch_tokens > tokens.numel():
            file_index = (file_index + 1) % len(files)
            tokens = load_shard(files[file_index])
            pos = 0

        windows = tokens[pos : pos + batch_tokens].view(batch_size, seq_len)
        pos += batch_tokens
        yield windows.to(device=DEVICE, dtype=torch.long)


@torch.no_grad()
def measure_sparsity(
    model: NemotronHForCausalLM,
    tracker: ActivationSparsityTracker,
    seq_len: int,
    batch_size: int,
    num_batches: int,
) -> None:
    """Run the model over FineWeb sequences and accumulate sparsity stats."""
    model.eval()

    # All sparsity hooks are in the backbone; skip vocabulary logits and loss.
    for inputs in sequence_batches(seq_len, batch_size, WARMUP_STEPS):
        model.model(input_ids=inputs, use_cache=False)

    tracker.reset()
    for inputs in sequence_batches(seq_len, batch_size, num_batches):
        model.model(input_ids=inputs, use_cache=False)


def analyze(
    *,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    num_batches: int = NUM_BATCHES,
) -> dict[str, float | int | str]:
    """Measure `num_batches` batches of `batch_size` sequences of `seq_len` tokens."""
    for name, value in (
        ("batch_size", batch_size),
        ("seq_len", seq_len),
        ("num_batches", num_batches),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if torch.device(DEVICE).type != "cuda" and os.environ.get("ALLOW_CPU") != "1":
        raise RuntimeError(
            "No usable CUDA device found (is the NVIDIA driver loaded?). Running an "
            "8B model on CPU is not practical; set ALLOW_CPU=1 to force it."
        )

    dtype = torch.bfloat16 if torch.device(DEVICE).type == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")

    print(f"model: {MODEL_NAME}")
    print(f"data: {DATA_ROOT / DATA_PATTERN}")
    model: NemotronHForCausalLM = cast(
        NemotronHForCausalLM,
        NemotronHForCausalLM.from_pretrained(
            MODEL_NAME, dtype=dtype, trust_remote_code=True, use_kernels=USE_KERNELS
        ).to(DEVICE),
    )

    # Plain dense forward so the forward hooks see the real activations.
    model.config.sparse_ffn = False
    model.config.use_ckpt = False

    tracker, handles = attach_sparsity_tracker(model)
    measure_sparsity(model, tracker, seq_len, batch_size, num_batches)

    result: dict[str, float | int | str] = {
        "model": MODEL_NAME,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "sequences": batch_size * num_batches,
        "avg_sparsity": tracker.average_overall(),
        "worst_sparsity": tracker.worst_overall(),
        "worst_batch_avg_sparsity": tracker.worst_batch_avg_as_float(),
    }
    worst = tracker.worst_as_floats()
    for layer_num, sparsity in tracker.average_as_floats().items():
        result[f"layer_{layer_num}_avg_sparsity"] = sparsity
        result[f"layer_{layer_num}_worst_sparsity"] = worst[layer_num]

    for handle in handles:
        handle.remove()
    del model
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.empty_cache()

    avg_sparsity = float(result["avg_sparsity"])
    worst_batch_avg = float(result["worst_batch_avg_sparsity"])
    print(
        f"  layers: {len(tracker.average)}, sequences: {result['sequences']}\n"
        f"  avg sparsity:    {avg_sparsity:.4f}\n"
        f"  worst batch avg: {worst_batch_avg:.4f}"
    )
    return result


def main() -> None:
    result = analyze()

    with RESULTS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"results: {RESULTS_PATH}")


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    main()
