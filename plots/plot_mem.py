"""Peak VRAM against input length for the language-model runs.

Each dataset is a table of measured peak VRAM, one column per configuration.
Every run is scattered and overlaid with a least-squares line of best fit, so the
rate at which activation memory grows with the input length can be compared
across the configurations. Runs stop at different input lengths, so a gap is
written as ``MISSING`` and the fit spans only what was actually measured.

Each fit is then extrapolated to the right edge of the plot, which keeps the
projected growth visible without the extrapolation enlarging either axis: the
axes are sized from the measured points, before any fit is drawn.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_lib import (
    CONFIG_STYLES, WIDE_FONT_SCALE, finish_plot, format_axes, plot_series,
)
from plot_tables import MIB_PER_GIB, parse_series, render


output_dir = Path(__file__).resolve().parent

YLABEL = "Peak VRAM / GiB"
# The plotted values are the harness readings rounded, so trailing zeros carry
# no information and are dropped.
YFORMAT = "{x:,.6g}"
# Shared by the scattered points and the legend symbols, which must match.
MARKER_SIZE = 5.5
MARKER_EDGE_WIDTH = 1.1

# Each dataset renders one figure. ``x_step`` pins the x ticks, which is what
# keeps their labels apart once the text is scaled up.
datasets = {
    "nemotron_mem.pdf": {
        "x_name": "N_input",
        "xlabel": "Input tokens",
        "x_step": 1000.0,
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
        "x_step": 5000.0,
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


def format_equation(slope):
    """Return a fitted line as a ``y = ax + C`` label.

    The slope is quoted per 1000 input tokens, which keeps its leading digits
    readable instead of a long run of zeros. Three decimals are kept because the
    optimised runs differ by less than that, so rounding to two would print the
    same equation for two different runs.

    The intercept collapses to ``C``: it captures the model's fixed footprint,
    which is much the same for every run and so adds nothing to the comparison.
    """
    return f"$y = {slope * 1000:.3f}x + C$"


def draw_fit(ax, x_values, y_values, *, color, x_max):
    """Draw a least-squares fit out to ``x_max`` and return its slope.

    The dotted line starts at the run's first measurement and runs to the right
    edge of the plot, so beyond the run's final point it is an extrapolation of
    the fitted trend rather than measured data.
    """
    slope, intercept = np.polyfit(x_values, y_values, 1)
    fit_x = np.array([min(x_values), x_max])
    ax.plot(fit_x, slope * fit_x + intercept, color=color, linestyle=":",
            linewidth=1.4)
    return slope


def equation_legend(fits):
    """Build legend rows pairing each run with the equation of its fit.

    The equation rows carry a blank handle, which indents them under the symbol
    they belong to.
    """
    handles, labels = [], []
    for name, equation in fits:
        style = CONFIG_STYLES[name]
        handles.append(Line2D(
            [], [], color=style["color"], marker=style["marker"],
            linestyle="none", markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH, markersize=MARKER_SIZE,
        ))
        labels.append(name)
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(equation)
    return handles, labels


def plot_mem(dataset):
    """Build one peak-VRAM figure and return it without saving it."""
    series = parse_series(
        dataset["table"], x_name=dataset["x_name"], y_scale=1 / MIB_PER_GIB,
    )

    fig, ax = plt.subplots()
    plot_series(
        ax, series, linestyle="none",
        markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
    )

    # The measured points alone decide the axes' extent. Capturing it here lets
    # the fits reach the right edge while their extrapolated values stay out of
    # the autoscaling, so the range reflects only what was measured.
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    fits = [
        (name, format_equation(draw_fit(
            ax, x_values, y_values, color=CONFIG_STYLES[name]["color"],
            x_max=x_limits[1],
        )))
        for name, x_values, y_values in series
    ]

    format_axes(
        ax, xlabel=dataset["xlabel"], ylabel=YLABEL, yformat=YFORMAT,
        x_step=dataset["x_step"],
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    finish_plot(ax, legend_outside=True, legend_entries=equation_legend(fits))
    return fig, ax


def main():
    render(
        datasets, plot_mem, output_dir=output_dir,
        wide=True, font_scale=WIDE_FONT_SCALE,
    )


if __name__ == "__main__":
    main()
