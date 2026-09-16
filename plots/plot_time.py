"""Wall-clock time against the number of tokens per step for the language-model runs.

Each dataset is a table of measured step times, one column per configuration.
Every run is drawn as its own line, so the cost of each configuration can be read
directly against the dense baseline. Runs stop at different token counts, so
a gap is written as ``MISSING`` and the line simply ends there.

Both figures are drawn in milliseconds. The harnesses disagree on units, so each
dataset carries the factor that brings its own times onto that shared axis.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import WIDE_FONT_SCALE, finish_plot, format_axes, plot_series
from plot_tables import parse_series, render


output_dir = Path(__file__).resolve().parent

YLABEL = "Time / ms"
# Both datasets record whole milliseconds, so integer ticks need no decimal.
YFORMAT = "{x:,.0f}"

# The nanoGPT harness reports seconds while the Nemotron harness reports
# milliseconds. Scaling at the point of use keeps each table faithful to its
# source while the two figures share one axis.
MS_PER_S = 1000.0

# Each dataset renders one figure. ``x_step`` pins the x ticks, which is what
# keeps their labels apart once the text is scaled up. ``x_scale`` converts the
# table's x column onto the plotted axis: the harnesses disagree on what an x
# value counts, so each dataset carries its own factor.
datasets = {
    "nemotron_time.pdf": {
        "x_name": "Length",
        # The Nemotron harness processes one sequence per step, so its sequence
        # length is already its token count.
        "xlabel": "Tokens",
        "y_scale": 1.0,
        "x_step": 1000.0,
        "table": """
Length Dense BitSparse BitSparse-sb Checkpoint
50 68 75 84 82
100 71 75 89 84
200 71 79 83 83
300 73 77 83 86
400 82 88 94 93
500 95 100 103 109
700 128 129 133 145
900 166 168 169 188
1100 198 199 200 244
1300 231 230 232 259
1500 265 266 268 298
2000 356 357 358 401
2500 444 445 447 502
3000 531 532 536 599
3500 624 626 630 705
4000 710 712 717 799
4500 805 809 810 905
5000 895 895 896 1003
5500 - 979 981 1101
6000 - 1065 1067 1201
6500 - 1154 1153 1305
7000 - 1233 1233 1406
7500 - - 1328 1512
""",
    },
    "nanogpt_time.pdf": {
        "x_name": "Length",
        # The nanoGPT harness packs four sequences into every step, so a step
        # covers four times the sequence length in tokens.
        "xlabel": "Tokens",
        "y_scale": MS_PER_S,
        "x_scale": 4.0,
        "x_step": 20000.0,
        "table": """
Length Dense BitSparse BitSparse-sb Checkpoint
256 0.022 0.025 0.025 0.025
512 0.038 0.038 0.039 0.043
1024 0.068 0.070 0.072 0.078
2048 0.135 0.135 0.137 0.156
4096 0.287 0.286 0.289 0.322
8192 0.686 0.684 0.691 0.758
12000 1.157 1.153 1.164 1.257
16000 - 1.755 1.767 1.896
""",
    },
}


def plot_time(dataset):
    """Build one elapsed-time figure and return it without saving it."""
    series = parse_series(
        dataset["table"], x_name=dataset["x_name"], y_scale=dataset["y_scale"],
    )

    x_scale = dataset.get("x_scale", 1.0)
    if x_scale != 1.0:
        series = [(name, [x * x_scale for x in xs], ys)
                  for name, xs, ys in series]
    fig, ax = plt.subplots()
    # One line per configuration, each solid with its own marker.
    plot_series(ax, series)
    format_axes(
        ax, xlabel=dataset["xlabel"], ylabel=YLABEL, yformat=YFORMAT,
        x_step=dataset["x_step"],
    )
    finish_plot(ax, legend_outside=True)
    return fig, ax


def main():
    render(
        datasets, plot_time, output_dir=output_dir,
        wide=True, font_scale=WIDE_FONT_SCALE,
    )


if __name__ == "__main__":
    main()
