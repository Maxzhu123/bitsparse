"""Peak backward-pass VRAM against input length for the language-model runs.

Each dataset is a table of measured peak VRAM, one column per configuration.
Every run is scattered and overlaid with a least-squares line of best fit, so the
rate at which activation memory grows with the input length can be compared
across the dense, sparsified, 15-bit and checkpointed configurations. Runs stop
at different input lengths, so unmeasured points are written as ``MISSING`` and
skipped. Every fit is then extrapolated out to the right edge of the plot, so the
projected growth of each run stays visible without the extrapolation itself
enlarging either axis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_lib import (
    CONFIG_SERIES_STYLES, finish_plot, format_axes, plot_series, plot_style,
)


output_dir = Path(__file__).resolve().parent

# ``nemotron.utils.print_max_memory`` and the nanoGPT harness both report MiB;
# the figures use GiB.
MIB_PER_GIB = 1024.0

# Written where a run was not measured at that input length. Every row must spell
# out its gaps, otherwise a column would silently shift into another run's name.
MISSING = "-"

# Each dataset renders one figure. Table headers become series names verbatim,
# except that underscores are read back as spaces; keeping them single tokens
# lets a table stay whitespace-separable with its gaps explicit.
datasets = {
    "nemotron_mem.pdf": {
        "x_name": "N_input",
        "xlabel": "Input tokens",
        "table": """
N_input Base BitSparse Sign-bit Checkpoint
50 16750 16660 16660 16652
100 17010 16828 16828 16813
200 17528 17165 17165 17135
300 18047 17498 17497 17457
400 18566 17835 17834 17779
500 19084 18168 18167 18101
700 20122 18839 18836 18744
900 21254 19607 19603 19482
1100 22540 20527 20522 20374
1300 23825 21445 21440 21266
1500 25111 22364 22357 22158
2000 28326 24662 24654 24388
2500 31540 26962 26951 26619
3000 34754 29256 29243 28849
3500 - 31554 31539 31079
4000 - 33853 33835 33309
4500 - 36149 36129 35539
5000 - - - 37770
""",
    },
    "nanogpt_mem.pdf": {
        "x_name": "Length",
        "xlabel": "Sequence length",
        "table": """
Length Base BitSparse Sign-bit Checkpoint
256 1297 1165 1164 1116
512 1688 1435 1427 1286
1024 2549 2047 2040 1853
2048 4254 3250 3239 2990
4096 7686 5663 5649 5270
8192 14553 10492 10468 9834
12000 20994 15017 14979 14104
16384 - 20128 20080 18958
""",
    },
}


def parse_data(table, *, x_name):
    """Return ``(label, x_values, y_values)`` per run, skipping ``MISSING`` cells."""
    rows = [line.split() for line in table.strip().splitlines() if line.strip()]
    if len(rows) < 2 or rows[0][0] != x_name:
        raise ValueError(f"Expected a {x_name} header and at least one data row")
    headers = rows[0]
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate run names")

    x_values = {name: [] for name in headers[1:]}
    y_values = {name: [] for name in headers[1:]}
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(headers):
            raise ValueError(f"Row {row_number}: expected {len(headers)} values, got {len(row)}")
        for name, value in zip(headers[1:], row[1:]):
            if value == MISSING:
                continue
            x_values[name].append(float(row[0]))
            y_values[name].append(float(value.replace(",", "")))

    return [
        (name.replace("_", " "), x_values[name], y_values[name])
        for name in headers[1:]
    ]


def format_fit_equation(slope):
    """Return a fitted line as a compact ``y = ax + C`` legend label.

    The slope is quoted per 1000 input tokens, which keeps its leading digits
    readable instead of a long run of zeros. Three decimals are kept because
    the sparsified runs differ by less than that (e.g. nanoGPT's 15-bit packing
    barely moves the slope), and rounding to two would print identical
    equations for two different runs. The intercept collapses to ``C``: it
    only captures the model's fixed footprint, which is the same for every run
    here and so adds nothing to the comparison.
    """
    return f"$y = {slope * 1000:.3f}x + C$"


def plot_fit(ax, x_values, y_values, *, color, x_max):
    """Overlay a least-squares fit, extrapolated out to ``x_max``.

    The dotted line starts at the run's first measurement and continues to
    ``x_max``, the right edge of the plot. Beyond the run's own final point the
    line is therefore an extrapolation of the fitted trend, not measured data,
    and it is clipped by the axes if the trend leaves the plotted y range.
    Return the fitted ``(slope, intercept)``.
    """
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_fit = np.array([min(x_values), x_max], dtype=float)
    ax.plot(
        x_fit, slope * x_fit + intercept,
        color=color, linestyle=":", linewidth=1.4,
    )
    return slope, intercept


def plot_mem(table, *, x_name, xlabel):
    """Build a VRAM/input-length figure and return it without saving it."""
    # The harnesses report MiB; scale to GiB for a more readable axis.
    series = [
        (label, x_values, [vram / MIB_PER_GIB for vram in y_values])
        for label, x_values, y_values in parse_data(table, x_name=x_name)
    ]
    with plot_style(wide=True):
        fig, ax = plt.subplots()
        plot_series(
            ax, series, styles=CONFIG_SERIES_STYLES, linestyle="none",
            markersize=5.5, markeredgewidth=1.1,
        )
        # The measured points alone decide the axes' extent. Capturing it here
        # lets the fits run out to the plot's right edge while the extrapolated
        # values stay out of the autoscaling, so the y range reflects only what
        # was actually measured.
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        # One dotted fit per run, drawn to the same right edge. Each run's
        # equation is kept so the legend can show it under that run's symbol.
        equations = []
        for (_, x_values, y_values), style in zip(series, CONFIG_SERIES_STYLES):
            slope, _ = plot_fit(
                ax, x_values, y_values, color=style["color"], x_max=x_limits[1],
            )
            equations.append(format_fit_equation(slope))
        format_axes(
            ax, xlabel=xlabel, ylabel="Peak VRAM (GiB)", yformat="{x:,.1f}",
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        # Build the legend by hand so each equation gets a row of its own
        # beneath its run's symbol. Those rows carry a blank handle, which
        # indents them under the symbol they belong to.
        legend_handles, legend_labels = [], []
        for (name, _, _), style, equation in zip(series, CONFIG_SERIES_STYLES, equations):
            legend_handles.append(Line2D(
                [], [], color=style["color"], marker=style["marker"],
                linestyle="none", markerfacecolor="white",
                markeredgewidth=1.1, markersize=5.5,
            ))
            legend_labels.append(name)
            legend_handles.append(Line2D([], [], linestyle="none"))
            legend_labels.append(equation)
        finish_plot(
            ax, legend_outside=True,
            legend_entries=(legend_handles, legend_labels),
        )
    return fig, ax


def main():
    # Save inside the style context so the configured ``savefig.bbox`` (tight)
    # and font settings apply; otherwise figures clip at the canvas edges.
    with plot_style():
        for filename, dataset in datasets.items():
            fig, _ = plot_mem(
                dataset["table"], x_name=dataset["x_name"], xlabel=dataset["xlabel"],
            )
            fig.savefig(output_dir / filename, format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
