"""Wall-clock time against input length for the language-model runs.

Each dataset is a table of measured step times, one column per configuration.
Every run is drawn as its own line, so the cost of each configuration can be read
directly against the dense baseline. Runs stop at different input lengths, so a
gap is written as ``MISSING`` and the line simply ends there.

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
# keeps their labels apart once the text is scaled up.
datasets = {
    "nemotron_time.pdf": {
        "x_name": "N_input",
        "xlabel": "Input tokens",
        "y_scale": 1.0,
        "x_step": 1000.0,
        "table": """
N_input Base BitSparse Sign-bit Checkpoint
50 185 185 198 197
100 187 185 202 200
200 186 186 203 198
300 189 189 205 201
400 188 194 208 199
500 189 194 207 199
700 219 229 225 253
900 283 290 291 328
1100 324 332 331 374
1300 387 396 394 447
1500 434 443 444 543
2000 567 580 579 656
2500 703 716 718 811
3000 851 868 869 986
3500 - 1020 1019 1159
4000 - 1142 1142 1294
4500 - 1300 1299 1475
""",
    },
    "nanogpt_time.pdf": {
        "x_name": "Length",
        "xlabel": "Sequence length",
        "y_scale": MS_PER_S,
        "x_step": 5000.0,
        "table": """
Length Base BitSparse Sign-bit Checkpoint
256 0.022 0.024 0.025 0.025
512 0.038 0.038 0.038 0.044
1024 0.070 0.070 0.071 0.079
2048 0.139 0.135 0.136 0.150
4096 0.294 0.285 0.287 0.313
8192 0.701 0.682 0.685 0.736
12000 1.183 1.151 1.161 1.234
16384 - 1.813 1.818 1.920
""",
    },
}


def plot_time(dataset):
    """Build one elapsed-time figure and return it without saving it."""
    series = parse_series(
        dataset["table"], x_name=dataset["x_name"], y_scale=dataset["y_scale"],
    )

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
