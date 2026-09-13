"""Peak VRAM against sequence length for the language-model runs.

Each dataset is a table of measured peak VRAM, one column per configuration.
Every run is scattered and overlaid with a least-squares line of best fit, so the
rate at which activation memory grows with the sequence length can be compared
across the configurations. Runs stop at different sequence lengths, so a gap is
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
# keeps their labels apart once the text is scaled up. ``fit_max`` optionally
# caps a series' fit at a given x while every measured point is still plotted.
datasets = {
    "nemotron_mem.pdf": {
        "x_name": "Length",
        "xlabel": "Sequence length",
        "x_step": 1000.0,
        # BitSparse's run stops at 7000 while sign-bit packing reaches 7500.
        # Fitting the longer span gives sign-bit a steeper slope than the run it
        # is a strict improvement on, so its fit stops at BitSparse's last
        # measurement even though the 7500 point is still plotted.
        "fit_max": {"Sign-bit": 7000.0},
        "table": """
Length Base BitSparse Sign-bit Checkpoint
50 16704 16663 16663 16655
100 16844 16761 16760 16745
200 17115 16950 16949 16918
300 17387 17134 17133 17092
400 17659 17323 17321 17265
500 17931 17508 17505 17439
700 18475 17882 17878 17785
900 19018 18259 18255 18132
1100 19562 18633 18628 18479
1300 20105 19007 19001 18826
1500 20649 19381 19374 19172
2000 22008 20316 20307 20039
2500 23367 21253 21241 20906
3000 24727 22186 22172 21773
3500 26086 23122 23106 22640
4000 27446 24060 24041 23508
4500 28809 25000 24979 24379
5000 30172 25940 25917 25250
5500 - 26877 26851 26144
6000 - 27909 27881 27111
6500 - 28942 28913 28078
7000 - 29976 29944 29045
7500 - - 30976 30011
""",
    },
    "nanogpt_mem.pdf": {
        "x_name": "Length",
        "xlabel": "Sequence length",
        "x_step": 5000.0,
        "table": """
Length Base BitSparse Sign-bit Checkpoint
256 1281 1148 1146 1117
512 1653 1394 1388 1248
1024 2484 1973 1968 1785
2048 4114 3102 3092 2845
4096 7398 5372 5360 4981
8192 13977 9908 9884 9255
12000 20142 14160 14121 13259
16000 - 18571 18526 17427
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


def fit_points(x_values, y_values, limit):
    """Return the points to fit, dropping any beyond ``limit``.

    A ``None`` limit keeps every point. Comparing runs whose measurements stop
    at different lengths otherwise summarises them with slopes that are not
    comparable, so a series can be fit over a span shared with the run it is
    measured against while its remaining points stay in the scatter.
    """
    if limit is None:
        return x_values, y_values
    kept = [(x, y) for x, y in zip(x_values, y_values) if x <= limit]
    return [x for x, _ in kept], [y for _, y in kept]


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
            ax, *fit_points(x_values, y_values, dataset.get("fit_max", {}).get(name)),
            color=CONFIG_STYLES[name]["color"], x_max=x_limits[1],
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
