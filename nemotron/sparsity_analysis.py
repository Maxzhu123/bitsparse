"""Measure Nemotron's retained ReLU values on re-tokenized FineWeb text.

Run from the repository root with `python -m nemotron.sparsity_analysis`.
Sparsity means the fraction of zero ReLU activations; density is reported
separately. Samples are spread across the validation shards, without wrapping.
"""
import csv
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import tiktoken
import torch
from torch import Tensor
from transformers import AutoTokenizer

from nemotron import llm

MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
RESULTS_PATH = Path(__file__).resolve().parent / "sparsity.csv"
DATA_ROOT = Path(os.environ.get("FINEWEB_ROOT", str(Path(__file__).resolve().parents[1] / "nanogpt/data")))
DATA_PATTERN = "fineweb10B/fineweb_val_*.bin"
SHARD_HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
BATCH_SIZE = 1
SEQ_LEN = 8000
NUM_BATCHES = 256
WARMUP_STEPS = 1


class ActivationSparsityTracker:
    """Count positive preactivations: exactly the values retained by bitsparse."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.reset()

    def reset(self) -> None:
        self.nonzeros: dict[int, Tensor] = {}
        self.elements: dict[int, int] = {}
        self.counts: dict[int, int] = {}
        self.worst: dict[int, Tensor] = {}
        self.batch_counts: dict[int, tuple[Tensor, int]] = {}
        self.batch_sparsities: list[Tensor] = []

    @torch.no_grad()
    def update(self, layer_num: int, preactivation: Tensor) -> None:
        count = (preactivation > 0).sum(dtype=torch.int64)
        size = preactivation.numel()
        sparsity = 1 - count.double() / size
        if layer_num not in self.nonzeros:
            self.nonzeros[layer_num] = count
            self.elements[layer_num] = size
            self.counts[layer_num] = 1
            self.worst[layer_num] = sparsity
        else:
            self.nonzeros[layer_num] += count
            self.elements[layer_num] += size
            self.counts[layer_num] += 1
            self.worst[layer_num] = torch.minimum(self.worst[layer_num], sparsity)
        if layer_num in self.batch_counts:
            raise RuntimeError("An MLP was counted twice in one forward pass")
        self.batch_counts[layer_num] = (count, size)
        if len(self.batch_counts) == self.num_layers:
            nonzeros = torch.stack([v[0] for v in self.batch_counts.values()]).sum()
            elements = sum(v[1] for v in self.batch_counts.values())
            self.batch_sparsities.append(1 - nonzeros.double() / elements)
            self.batch_counts.clear()

    def average_as_floats(self) -> dict[int, float]:
        return {layer: 1 - count.item() / self.elements[layer]
                for layer, count in sorted(self.nonzeros.items())}

    def worst_as_floats(self) -> dict[int, float]:
        return {layer: value.item() for layer, value in sorted(self.worst.items())}

    def average_overall(self) -> float:
        return 1 - sum(v.item() for v in self.nonzeros.values()) / sum(self.elements.values())

    def worst_overall(self) -> float:
        return min(self.worst_as_floats().values())

    def worst_batch_avg_as_float(self) -> float:
        return torch.stack(self.batch_sparsities).min().item()


@contextmanager
def attach_sparsity_tracker(model: llm.NemotronHForCausalLM):
    """Observe the fused dense FFN without recomputing its projection.

    up_proj and act_fn module hooks do not see this path: the FFN uses
    FusedRMSNormMLP and _DenseRelu2Linear autograd Functions directly.
    Restore the temporary observer even if model execution fails.
    """
    indices = [i for i, block in enumerate(model.model.layers) if block.block_type == "mlp"]
    if not indices:
        raise ValueError("Model contains no MLP blocks")
    tracker = ActivationSparsityTracker(len(indices))
    current_layer = None
    handles = []
    original = llm._DenseRelu2Linear.forward

    def select_layer(index):
        def hook(module, inputs):
            nonlocal current_layer
            current_layer = index
        return hook

    def observe(ctx, z, down_weight, down_bias):
        tracker.update(current_layer, z)
        return original(ctx, z, down_weight, down_bias)

    try:
        for index in indices:
            handles.append(model.model.layers[index].mixer.register_forward_pre_hook(select_layer(index)))
        llm._DenseRelu2Linear.forward = staticmethod(observe)
        yield tracker
    finally:
        llm._DenseRelu2Linear.forward = staticmethod(original)
        for handle in handles:
            handle.remove()


def load_shard(path: Path) -> Tensor:
    """Memory-map the GPT-2-tokenized validation shard on CPU."""
    header = torch.from_file(str(path), False, SHARD_HEADER_INTS, dtype=torch.int32)
    if int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Invalid FineWeb shard header: {path}")
    num_tokens = int(header[2])
    header_tokens = SHARD_HEADER_INTS * 2
    if path.stat().st_size != (num_tokens + header_tokens) * 2:
        raise ValueError(f"Invalid shard size: {path}")
    return torch.from_file(str(path), False, num_tokens + header_tokens, dtype=torch.uint16)[header_tokens:]


def sequence_batches(tokenizer, seq_len: int, batch_size: int, num_batches: int) -> Iterator[Tensor]:
    """Decode GPT-2 IDs, then encode text using Nemotron's own vocabulary."""
    files = sorted(DATA_ROOT.glob(DATA_PATTERN))
    if not files:
        raise FileNotFoundError(f"No shards matched {DATA_ROOT / DATA_PATTERN}")
    source_tokenizer = tiktoken.get_encoding("gpt2")
    shards = [load_shard(path) for path in files]
    sequences = batch_size * num_batches
    source_length = 4 * seq_len  # Enough text to fill a Nemotron window.
    batch = []
    for index in range(sequences):
        shard_index = index % len(shards)
        shard = shards[shard_index]
        in_shard = index // len(shards)
        samples_in_shard = (sequences - 1 - shard_index) // len(shards) + 1
        stride = (shard.numel() - source_length) // max(samples_in_shard - 1, 1)
        if stride < source_length:
            raise ValueError("Not enough validation text for non-overlapping samples")
        start = in_shard * stride
        text = source_tokenizer.decode(shard[start:start + source_length].tolist())
        text = text.replace(source_tokenizer.decode([source_tokenizer.eot_token]), tokenizer.eos_token)
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) != seq_len:
            raise ValueError("Decoded source window is too short after re-tokenization")
        batch.append(ids)
        if len(batch) == batch_size:
            yield torch.tensor(batch, device="cuda", dtype=torch.long)
            batch.clear()


@torch.no_grad()
def measure_sparsity(model, tracker, tokenizer, seq_len, batch_size, num_batches):
    model.eval()
    batches = sequence_batches(tokenizer, seq_len, batch_size, num_batches + WARMUP_STEPS)
    for _ in range(WARMUP_STEPS):
        model.model(input_ids=next(batches), use_cache=False)
    tracker.reset()
    for index, inputs in enumerate(batches, 1):
        model.model(input_ids=inputs, use_cache=False)
        print(f"batch {index}/{num_batches}: zero fraction {tracker.batch_sparsities[-1].item():.4%}", flush=True)
    if len(tracker.batch_sparsities) != num_batches or any(n != num_batches for n in tracker.counts.values()):
        raise RuntimeError("Incomplete activation measurements")


def analyze(*, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_batches=NUM_BATCHES):
    for name, value in (("batch_size", batch_size), ("seq_len", seq_len), ("num_batches", num_batches)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError("Nemotron sparsity analysis requires CUDA")
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"model: {MODEL_NAME}\ndata: {DATA_ROOT / DATA_PATTERN}", flush=True)
    model = llm.NemotronHForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, local_files_only=True,
    ).cuda()
    model.config.sparse_ffn = False
    model.config.use_ckpt = False
    try:
        with attach_sparsity_tracker(model) as tracker:
            measure_sparsity(model, tracker, tokenizer, seq_len, batch_size, num_batches)
        average = tracker.average_overall()
        result = dict(model=MODEL_NAME, seq_len=seq_len, batch_size=batch_size,
                      num_batches=num_batches, sequences=batch_size*num_batches,
                      tokens=seq_len*batch_size*num_batches,
                      source_tokenizer="gpt2", model_tokenizer=MODEL_NAME,
                      avg_sparsity=average, avg_nonzero_fraction=1-average,
                      worst_sparsity=tracker.worst_overall(),
                      worst_batch_avg_sparsity=tracker.worst_batch_avg_as_float())
        worst = tracker.worst_as_floats()
        for layer, sparsity in tracker.average_as_floats().items():
            result[f"layer_{layer}_avg_sparsity"] = sparsity
            result[f"layer_{layer}_worst_sparsity"] = worst[layer]
            print(f"layer {layer:2d}: {sparsity:.4%} zeros", flush=True)
        print(f"Average sparsity: {average:.4%}; nonzeros: {1-average:.4%}", flush=True)
        print(f"Densest batch sparsity: {result['worst_batch_avg_sparsity']:.4%}", flush=True)
        return result
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    result = analyze()
    with RESULTS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
