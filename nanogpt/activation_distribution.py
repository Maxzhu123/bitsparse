from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn
from tqdm import trange

from dataloader import data_generator
from histogram import plot_histogram, tensor_histogram
from nanogpt import CausalSelfAttention, GPT, Linear, MLP, RMSNorm


CHECKPOINT_PATH = Path(__file__).parent / "logs/2026-07-04_00-06-23/3300.pt"
DATA_PATTERN = "data/fineweb10B/fineweb_val_*.bin"
NUM_BATCHES = 4
SEQUENCE_LENGTH = 1024
SEQUENCES_PER_BATCH = 4
BIN_WIDTH = 0.5
LIMIT = 100.0
LAYER_TYPES = {
    "Embedding outputs": nn.Embedding,
    "Linear outputs": Linear,
    "RMSNorm outputs": RMSNorm,
    "Attention outputs": CausalSelfAttention,
    "MLP outputs": MLP,
}


def get_state_dict(checkpoint_path: Path) -> dict[str, Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{checkpoint_path} does not contain a state_dict")
    return {
        name.removeprefix("_orig_mod."): tensor
        for name, tensor in checkpoint.items()
    }


def infer_model_config(state_dict: dict[str, Tensor]) -> tuple[int, int, int]:
    vocab_size, model_dim = state_dict["embed.weight"].shape
    block_ids = {
        int(name.split(".", 2)[1])
        for name in state_dict
        if name.startswith("blocks.")
    }
    if not block_ids:
        raise ValueError("Could not infer the number of layers from the checkpoint")
    return vocab_size, max(block_ids) + 1, model_dim


@torch.inference_mode()
def activation_distribution(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    layer_types: dict[str, type[nn.Module]],
    *,
    num_batches: int,
    bin_width: float = BIN_WIDTH,
    limit: float = LIMIT,
) -> tuple[dict[str, Tensor], Tensor, dict[str, tuple[Tensor, Tensor]]]:
    """Aggregate output histograms for several module types in one pass."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if not layer_types:
        raise ValueError("layer_types must not be empty")

    counts: dict[str, Tensor | None] = {
        label: None for label in layer_types
    }
    minima: dict[str, Tensor | None] = {
        label: None for label in layer_types
    }
    maxima: dict[str, Tensor | None] = {
        label: None for label in layer_types
    }
    edges: Tensor | None = None
    modules = list(model.modules())
    matched_modules = {
        label: sum(isinstance(module, layer_type) for module in modules)
        for label, layer_type in layer_types.items()
    }

    def activation_hook(label: str):
        def record_activation(
            module: nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            nonlocal edges
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Expected {type(module).__name__} to return a tensor, "
                    f"got {type(output).__name__}"
                )
            batch_counts, batch_edges, batch_minimum, batch_maximum = tensor_histogram(
                [output],
                bin_width=bin_width,
                limit=limit,
            )
            current_counts = counts[label]
            counts[label] = (
                batch_counts
                if current_counts is None
                else current_counts + batch_counts
            )
            current_minimum = minima[label]
            current_maximum = maxima[label]
            minima[label] = (
                batch_minimum
                if current_minimum is None
                else torch.minimum(current_minimum, batch_minimum)
            )
            maxima[label] = (
                batch_maximum
                if current_maximum is None
                else torch.maximum(current_maximum, batch_maximum)
            )
            edges = batch_edges

        return record_activation

    unmatched_labels = [
        label for label, num_matches in matched_modules.items() if num_matches == 0
    ]
    if unmatched_labels:
        raise ValueError(f"Model contains no matching modules for {unmatched_labels}")

    handles = []
    for module in modules:
        for label, layer_type in layer_types.items():
            if isinstance(module, layer_type):
                handles.append(module.register_forward_hook(activation_hook(label)))

    model.eval()
    try:
        iterator = iter(batches)
        for _ in trange(num_batches, desc="Collecting activations", unit="batch"):
            inputs, targets = next(iterator)
            model(inputs, targets)
    finally:
        for handle in handles:
            handle.remove()

    if (
        edges is None
        or any(value is None for value in counts.values())
        or any(value is None for value in minima.values())
        or any(value is None for value in maxima.values())
    ):
        raise RuntimeError("One or more activation hooks did not capture any outputs")
    histograms = {
        label: value
        for label, value in counts.items()
        if value is not None
    }
    extrema = {
        label: (minima[label], maxima[label])
        for label in layer_types
        if minima[label] is not None and maxima[label] is not None
    }
    return histograms, edges, extrema


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = get_state_dict(CHECKPOINT_PATH)
    vocab_size, num_layers, model_dim = infer_model_config(state_dict)
    model = GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        cfg={"bitsparse": False, "pack_sbit": False, "checkpoint": False},
    )
    model.load_state_dict(state_dict)
    model.to(device)

    batches = data_generator(
        DATA_PATTERN,
        SEQUENCES_PER_BATCH * SEQUENCE_LENGTH,
        seq_len=SEQUENCE_LENGTH,
        device=device,
        data_root=Path(__file__).parent,
    )
    histograms, edges, extrema = activation_distribution(
        model,
        batches,
        LAYER_TYPES,
        num_batches=NUM_BATCHES,
    )
    for label, (minimum, maximum) in extrema.items():
        print(f"{label}: min={minimum.item():.6g}, max={maximum.item():.6g}")
    plot_histogram(
        histograms,
        edges,
        log_y=True,
        title=f"NanoGPT activations over {NUM_BATCHES} batches",
    )


if __name__ == "__main__":
    main()
