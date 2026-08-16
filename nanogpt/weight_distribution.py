from pathlib import Path

import torch

from histogram import plot_histogram, tensor_histogram


checkpoint_path = Path(__file__).parents[1] / "nanogpt/logs/2026-07-04_00-06-23/3300.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = {
    name.removeprefix("_orig_mod."): tensor
    for name, tensor in checkpoint.get("model", checkpoint).items()
}

groups = {"Linear weights": [], "Biases": [], "Other": []}
for name, tensor in state_dict.items():
    module_name, _, value_name = name.rpartition(".")
    if (
        value_name == "weight"
        and tensor.ndim == 2
        and f"{module_name}.bias" in state_dict
    ):
        groups["Linear weights"].append(tensor)
    elif value_name == "bias":
        groups["Biases"].append(tensor)
    else:
        print(f'{name = }')
        groups["Other"].append(tensor)

histograms = {}
for label, tensors in groups.items():
    histograms[label], edges, _, _ = tensor_histogram(
        tensors,
        limit=50,
        bin_width=0.5,
    )

plot_histogram(
    histograms,
    edges,
    log_y=True,
    title="NanoGPT checkpoint tensor distributions",
)
