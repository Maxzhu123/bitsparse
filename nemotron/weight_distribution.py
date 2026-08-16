import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from histogram import plot_histogram, tensor_histogram
from nemotron.llm import NemotronHForCausalLM


MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
BIN_WIDTH = 0.5
LIMIT = 50.0


def group_parameters(model: nn.Module) -> dict[str, list[Tensor]]:
    """Group matrix/kernel multiplication parameters independently of module type."""
    groups: defaultdict[str, list[Tensor]] = defaultdict(list)
    seen: set[int] = set()
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    linear_module_weight_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.weight is not None
    }
    multiplication_weight_ids = linear_module_weight_ids | {
        id(parameter)
        for parameter in model.parameters()
        if parameter.ndim >= 2 and id(parameter) not in embedding_parameter_ids
    }

    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if id(parameter) in multiplication_weight_ids:
                label = "Linear weights"
            elif name == "bias":
                label = "Biases"
            else:
                label = "Other"
            groups[label].append(parameter)

    return {
        label: groups[label]
        for label in ("Linear weights", "Biases", "Other")
        if groups[label]
    }


@torch.no_grad()
def weight_distribution(
    model: nn.Module,
    *,
    bin_width: float = BIN_WIDTH,
    limit: float = LIMIT,
) -> tuple[dict[str, Tensor], Tensor, dict[str, tuple[Tensor, Tensor]]]:
    """Calculate histograms and extrema for Nemotron parameter groups."""
    groups = group_parameters(model)
    if not groups:
        raise ValueError("Model contains no parameters")

    histograms: dict[str, Tensor] = {}
    extrema: dict[str, tuple[Tensor, Tensor]] = {}
    edges: Tensor | None = None
    for label, tensors in groups.items():
        counts, group_edges, minimum, maximum = tensor_histogram(
            tensors,
            bin_width=bin_width,
            limit=limit,
        )
        histograms[label] = counts
        extrema[label] = (minimum, maximum)
        edges = group_edges

    if edges is None:
        raise RuntimeError("No parameter histograms were produced")
    return histograms, edges, extrema


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    ).to(device)

    histograms, edges, extrema = weight_distribution(model)
    for label, (minimum, maximum) in extrema.items():
        print(f"{label}: min={minimum.item():.6g}, max={maximum.item():.6g}")
    plot_histogram(
        histograms,
        edges,
        log_y=True,
        title="Nemotron checkpoint tensor distributions",
    )


if __name__ == "__main__":
    main()
