import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from tqdm import trange
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from histogram import tensor_histogram
from nemotron.llm import (
    NemotronHAttention,
    NemotronHForCausalLM,
    NemotronHMLP,
    NemotronHMamba2Mixer,
    NemotronHRMSNorm,
)


MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
SAMPLE_TEXT_PATH = Path(__file__).parent / "sample_text.txt"
NUM_BATCHES = 4
SEQUENCE_LENGTH = 1024
SEQUENCES_PER_BATCH = 1
BIN_WIDTH = 0.05
LIMIT = 150.0
BULK_MASS = 0.90
LAYER_TYPES = {
    "Linear outputs": nn.Linear,
    "Mamba outputs": NemotronHMamba2Mixer,
    "Attention outputs": NemotronHAttention,
    "MLP outputs": NemotronHMLP,
}


@dataclass(frozen=True)
class DistributionFit:
    family: str
    region: str
    center: float
    intercept: float
    slope: float
    radial_scale: float
    boundary: float
    log_rmse: float
    r_squared: float
    num_bins: int

    @property
    def parameters(self) -> str:
        if self.family == "gaussian":
            sigma = (-0.5 / self.slope) ** 0.5
            return f"mu={self.center:.4g}, sigma={sigma:.4g}"
        if self.family == "exponential":
            scale = -1.0 / self.slope
            return f"mu={self.center:.4g}, scale={scale:.4g}"
        exponent = -self.slope
        return (
            f"mu={self.center:.4g}, scale={self.radial_scale:.4g}, "
            f"exponent={exponent:.4g}"
        )

    def log_density(self, values: Tensor) -> Tensor:
        distance = (values - self.center).abs()
        if self.family == "gaussian":
            feature = distance.square()
        elif self.family == "exponential":
            feature = distance
        else:
            feature = torch.log1p(distance / self.radial_scale)
        return self.intercept + self.slope * feature


def _first_tensor(value: Any) -> Tensor | None:
    """Find the first tensor in a module output, including tuple outputs."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


@torch.inference_mode()
def activation_distribution(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor] | Tensor],
    layer_types: Mapping[str, type[nn.Module]],
    *,
    num_batches: int,
    bin_width: float = BIN_WIDTH,
    limit: float = LIMIT,
) -> tuple[dict[str, Tensor], Tensor, dict[str, tuple[Tensor, Tensor]]]:
    """Aggregate output histograms for several Nemotron module types."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if not layer_types:
        raise ValueError("layer_types must not be empty")

    counts: dict[str, Tensor | None] = {label: None for label in layer_types}
    minima: dict[str, Tensor | None] = {label: None for label in layer_types}
    maxima: dict[str, Tensor | None] = {label: None for label in layer_types}
    edges: Tensor | None = None
    modules = list(model.modules())

    unmatched_labels = [
        label
        for label, layer_type in layer_types.items()
        if not any(isinstance(module, layer_type) for module in modules)
    ]
    if unmatched_labels:
        raise ValueError(f"Model contains no matching modules for {unmatched_labels}")

    def activation_hook(label: str):
        def record_activation(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            nonlocal edges
            tensor = _first_tensor(output)
            if tensor is None:
                raise TypeError(
                    f"Expected {type(module).__name__} to return a tensor, "
                    f"got {type(output).__name__}"
                )
            batch_counts, batch_edges, batch_minimum, batch_maximum = tensor_histogram(
                [tensor],
                bin_width=bin_width,
                limit=limit,
            )
            counts[label] = (
                batch_counts
                if counts[label] is None
                else counts[label] + batch_counts
            )
            minima[label] = (
                batch_minimum
                if minima[label] is None
                else torch.minimum(minima[label], batch_minimum)
            )
            maxima[label] = (
                batch_maximum
                if maxima[label] is None
                else torch.maximum(maxima[label], batch_maximum)
            )
            edges = batch_edges

        return record_activation

    handles = []
    for module in modules:
        for label, layer_type in layer_types.items():
            if isinstance(module, layer_type):
                handles.append(module.register_forward_hook(activation_hook(label)))

    model.eval()
    try:
        iterator = iter(batches)
        for _ in trange(num_batches, desc="Collecting activations", unit="batch"):
            batch = next(iterator)
            if isinstance(batch, Mapping):
                model(**batch, use_cache=False)
            else:
                model(batch, use_cache=False)
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

    histograms = {label: value for label, value in counts.items() if value is not None}
    extrema = {
        label: (minima[label], maxima[label])
        for label in layer_types
        if minima[label] is not None and maxima[label] is not None
    }
    return histograms, edges, extrema


def _weighted_quantile(values: Tensor, weights: Tensor, quantile: float) -> float:
    order = torch.argsort(values)
    sorted_values = values[order]
    cumulative = weights[order].cumsum(0)
    target = quantile * cumulative[-1]
    index = int(torch.searchsorted(cumulative, target).clamp_max(len(values) - 1))
    return float(sorted_values[index])


def _fit_log_linear(
    feature: Tensor,
    log_density: Tensor,
    counts: Tensor,
) -> tuple[float, float, float, float] | None:
    """Fit log(density) = intercept + slope * feature."""
    # For Poisson histogram counts, Var(log(count)) is approximately 1/count,
    # so count weighting is the inverse-variance choice. It also prevents a
    # large number of noisy far-tail bins from lifting the fitted curve.
    weights = counts
    weight_sum = weights.sum()
    feature_mean = (weights * feature).sum() / weight_sum
    density_mean = (weights * log_density).sum() / weight_sum
    centered_feature = feature - feature_mean
    denominator = (weights * centered_feature.square()).sum()
    if denominator <= 0:
        return None

    slope = (weights * centered_feature * (log_density - density_mean)).sum() / denominator
    if slope >= 0:
        return None
    intercept = density_mean - slope * feature_mean
    residual = log_density - (intercept + slope * feature)
    squared_error = (weights * residual.square()).sum()
    total_error = (weights * (log_density - density_mean).square()).sum()
    log_rmse = torch.sqrt(squared_error / weight_sum)
    r_squared = 1.0 - squared_error / total_error if total_error > 0 else torch.nan
    return (
        float(intercept),
        float(slope),
        float(log_rmse),
        float(r_squared),
    )


def fit_distribution_regions(
    counts: Tensor,
    edges: Tensor,
    *,
    bulk_mass: float = BULK_MASS,
) -> dict[str, list[DistributionFit]]:
    """Fit Gaussian, exponential, and power-law models to bulk and tails."""
    if not 0.0 < bulk_mass < 1.0:
        raise ValueError("bulk_mass must be between zero and one")

    counts = counts[1:-1].detach().cpu().to(torch.float64)
    edges = edges.detach().cpu().to(torch.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    positive = counts > 0
    if positive.sum() < 6:
        raise ValueError("At least six populated histogram bins are required")

    centers = centers[positive]
    counts = counts[positive]
    bin_width = float(edges[1] - edges[0])
    density = counts / (counts.sum() * bin_width)
    log_density = density.log()
    center = _weighted_quantile(centers, counts, 0.5)
    distance = (centers - center).abs()
    boundary = _weighted_quantile(distance, counts, bulk_mass)
    radial_scale = max(
        _weighted_quantile(distance, counts, 0.5),
        bin_width,
    )

    region_masks = {
        "bulk": distance <= boundary,
        "tail": distance > boundary,
    }
    results: dict[str, list[DistributionFit]] = {}
    for region, mask in region_masks.items():
        if mask.sum() < 3:
            raise ValueError(
                f"The {region} region has fewer than three populated bins; "
                "reduce BIN_WIDTH or BULK_MASS"
            )
        region_distance = distance[mask]
        features = {
            "gaussian": region_distance.square(),
            "exponential": region_distance,
            "polynomial": torch.log1p(region_distance / radial_scale),
        }
        fits = []
        for family, feature in features.items():
            fitted = _fit_log_linear(feature, log_density[mask], counts[mask])
            if fitted is None:
                continue
            intercept, slope, log_rmse, r_squared = fitted
            fits.append(
                DistributionFit(
                    family=family,
                    region=region,
                    center=center,
                    intercept=intercept,
                    slope=slope,
                    radial_scale=radial_scale,
                    boundary=boundary,
                    log_rmse=log_rmse,
                    r_squared=r_squared,
                    num_bins=int(mask.sum()),
                )
            )
        if not fits:
            raise RuntimeError(f"No decaying model could be fit to the {region}")
        results[region] = sorted(fits, key=lambda fit: fit.log_rmse)
    return results


def fit_activation_distributions(
    histograms: Mapping[str, Tensor],
    edges: Tensor,
    *,
    bulk_mass: float = BULK_MASS,
) -> dict[str, dict[str, list[DistributionFit]]]:
    return {
        label: fit_distribution_regions(counts, edges, bulk_mass=bulk_mass)
        for label, counts in histograms.items()
    }


def print_fit_report(
    fits: Mapping[str, Mapping[str, list[DistributionFit]]],
) -> None:
    for label, regions in fits.items():
        print(f"\n{label}")
        for region, candidates in regions.items():
            best = candidates[0]
            print(
                f"  {region} (|x - mu| "
                f"{'<=' if region == 'bulk' else '>'} {best.boundary:.4g}):"
            )
            for fit in candidates:
                marker = "best" if fit is best else "    "
                print(
                    f"    {marker:>4} {fit.family:<11} "
                    f"log-RMSE={fit.log_rmse:.4f}, R^2={fit.r_squared:.4f}, "
                    f"{fit.parameters}"
                )


def plot_activation_fits(
    histograms: Mapping[str, Tensor],
    edges: Tensor,
    fits: Mapping[str, Mapping[str, list[DistributionFit]]],
) -> None:
    edges = edges.detach().cpu().to(torch.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_width = float(edges[1] - edges[0])
    figure, axes = plt.subplots(
        len(histograms),
        1,
        figsize=(12, 4 * len(histograms)),
        sharex=True,
        squeeze=False,
    )
    for axis, (label, raw_counts) in zip(axes[:, 0], histograms.items()):
        counts = raw_counts[1:-1].detach().cpu().to(torch.float64)
        density = counts / (counts.sum() * bin_width)
        positive = counts > 0
        axis.semilogy(centers[positive], density[positive], label="observed")

        for region, color in (("bulk", "tab:orange"), ("tail", "tab:red")):
            fit = fits[label][region][0]
            distance = (centers - fit.center).abs()
            masks = (
                [distance <= fit.boundary]
                if region == "bulk"
                else [
                    centers < fit.center - fit.boundary,
                    centers > fit.center + fit.boundary,
                ]
            )
            for index, mask in enumerate(masks):
                prediction = fit.log_density(centers[mask]).exp()
                axis.semilogy(
                    centers[mask],
                    prediction,
                    color=color,
                    linewidth=2,
                    label=f"{region}: {fit.family}" if index == 0 else None,
                )

        axis.set_title(label)
        axis.set_ylabel("Density")
        axis.legend()

    axes[-1, 0].set_xlabel("Activation")
    figure.suptitle(f"Nemotron activation fits (central mass={BULK_MASS:.0%})")
    figure.tight_layout()
    plt.show()


def token_batches(
    tokenizer,
    text: str,
    *,
    num_batches: int,
    sequence_length: int,
    sequences_per_batch: int,
    device: torch.device,
) -> Iterable[dict[str, Tensor]]:
    """Tokenize text and yield fixed-size, non-overlapping input batches."""
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].squeeze(0)
    batch_tokens = sequence_length * sequences_per_batch
    required_tokens = num_batches * batch_tokens
    if tokens.numel() < required_tokens:
        raise ValueError(
            f"Sample text contains {tokens.numel()} tokens; "
            f"{required_tokens} are required"
        )

    for start in range(0, required_tokens, batch_tokens):
        input_ids = tokens[start : start + batch_tokens].reshape(
            sequences_per_batch, sequence_length
        )
        yield {
            "input_ids": input_ids.to(device),
            "attention_mask": torch.ones_like(input_ids, device=device),
        }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    ).to(device)

    batches = token_batches(
        tokenizer,
        SAMPLE_TEXT_PATH.read_text(),
        num_batches=NUM_BATCHES,
        sequence_length=SEQUENCE_LENGTH,
        sequences_per_batch=SEQUENCES_PER_BATCH,
        device=device,
    )
    histograms, edges, extrema = activation_distribution(
        model,
        batches,
        LAYER_TYPES,
        num_batches=NUM_BATCHES,
    )
    for label, (minimum, maximum) in extrema.items():
        counts = histograms[label]
        overflow_fraction = float((counts[0] + counts[-1]) / counts.sum())
        print(
            f"{label}: min={minimum.item():.6g}, max={maximum.item():.6g}, "
            f"outside fit range={overflow_fraction:.3%}"
        )
    fits = fit_activation_distributions(histograms, edges)
    print_fit_report(fits)
    plot_activation_fits(histograms, edges, fits)


if __name__ == "__main__":
    main()
