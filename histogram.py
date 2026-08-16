import torch
from typing import Sequence
from matplotlib import pyplot as plt


MAX_CHUNK_ELEMENTS = 16 * 1024 * 1024


@torch.no_grad()
def tensor_histogram(
    tensors: Sequence[torch.Tensor],
    bin_width: float = 0.1,
    limit: float = 10.0,
):
    """
    Histogram with:
        bucket 0:          x < -limit
        buckets 1..N:     width `bin_width` from -limit to +limit
        bucket N+1:        x > +limit

    Returns:
        counts: [N + 2] int64 tensor
        edges:  [N + 1] bin edges for the normal buckets
        minimum, maximum: scalar tensors containing the overall extrema

    Assumes all tensors are on the same device.
    """
    if not tensors:
        raise ValueError("tensors must not be empty")

    device = tensors[0].device
    n_bins = round(2 * limit / bin_width)

    # +2 for negative and positive overflow buckets
    counts = torch.zeros(n_bins + 2, dtype=torch.int64, device=device)
    minimum = None
    maximum = None

    for tensor in tensors:
        tensor = tensor.detach().reshape(-1)
        if tensor.numel() == 0:
            continue

        tensor_minimum = tensor.amin()
        tensor_maximum = tensor.amax()
        minimum = (
            tensor_minimum
            if minimum is None
            else torch.minimum(minimum, tensor_minimum)
        )
        maximum = (
            tensor_maximum
            if maximum is None
            else torch.maximum(maximum, tensor_maximum)
        )

        # Keep the temporary int64 bucket indices bounded for very large model
        # tensors (for example Nemotron's embedding and output matrices).
        for start in range(0, tensor.numel(), MAX_CHUNK_ELEMENTS):
            x = tensor[start : start + MAX_CHUNK_ELEMENTS]
            # Histogram arithmetic needs more precision than model activations.
            # In BF16, adding a large limit (for example 150) can quantize away
            # bin widths smaller than one before the division is performed.
            bucket_values = x.to(torch.float32)

            # Normal buckets:
            # [-10.0, -9.9) -> 1
            # [-9.9,  -9.8) -> 2
            # ...
            idx = torch.floor((bucket_values + limit) / bin_width).to(torch.int64) + 1

            # Overflow buckets
            idx = torch.where(bucket_values < -limit, 0, idx)
            idx = torch.where(bucket_values > limit, n_bins + 1, idx)

            # x == +limit goes into the final regular bucket
            idx.clamp_(0, n_bins + 1)

            counts += torch.bincount(idx, minlength=n_bins + 2)

    edges = torch.linspace(
        -limit, limit, n_bins + 1,
        device=device,
    )

    if minimum is None or maximum is None:
        raise ValueError("tensors must contain at least one value")
    return counts, edges, minimum, maximum

def plot_histogram(counts, edges, log_y=False, title="Tensor value histograms"):
    edges = edges.detach().cpu()
    if torch.is_tensor(counts):
        counts = {"Values": counts}
    counts = {
        label: values.detach().cpu()
        for label, values in counts.items()
    }

    lower_limit = edges[0].item()
    upper_limit = edges[-1].item()
    figure, axes = plt.subplots(
        len(counts),
        1,
        figsize=(12, 4 * len(counts)),
        sharex=True,
        squeeze=False,
    )
    for axis, (label, values) in zip(axes[:, 0], counts.items()):
        axis.stairs(values[1:-1].numpy(), edges.numpy())
        axis.set_title(label)
        axis.set_ylabel("Count")
        if log_y:
            axis.set_yscale("log")
        axis.text(
            0.01,
            0.95,
            f"x < {lower_limit:g}: {values[0]:,}\n"
            f"x > {upper_limit:g}: {values[-1]:,}",
            transform=axis.transAxes,
            va="top",
        )

    axes[-1, 0].set_xlabel("Value")
    figure.suptitle(title)

    plt.tight_layout()
    plt.show()
