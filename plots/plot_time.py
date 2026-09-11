"""Wall-clock time against input length for the language-model runs.
Each dataset is a table of measured step times, one column per configuration.
Every run is drawn as its own line, so the cost of the sparsified, packed and
checkpointed variants can be read directly against the dense baseline. Runs stop
at different input lengths, so unmeasured points are written as ``MISSING`` and
the line simply ends there.

Both figures are drawn in milliseconds. The harnesses disagree on units, so each
dataset carries the factor that brings its own times onto that shared axis.
"""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

from plot_lib import (
    CONFIG_SERIES_STYLES, finish_plot, format_axes, plot_series, plot_style,
)


output_dir = Path(__file__).resolve().parent

# Written where a run was not measured at that input length. Every row must spell
# out its gaps, otherwise a column would silently shift into another run's name.
MISSING = "-"

# The nanoGPT harness reports seconds; the Nemotron harness already reports
# milliseconds. Scaling here keeps each table faithful to its source while the
# two figures share one axis.
MS_PER_S = 1000.0
YLABEL = "Time / ms"

# Each dataset renders one figure. Table headers become series names verbatim,
# except that underscores are read back as spaces; keeping them single tokens
# lets a table stay whitespace-separable with its gaps explicit.


class Dataset(TypedDict):
    """One figure: the x column name, its axis title, and the time-unit factor."""

    x_name: str
    xlabel: str
    y_scale: float
    table: str


datasets: dict[str, Dataset] = {
    "nemotron_time.pdf": {
        "x_name": "N_input",
        "xlabel": "Input tokens",
        "y_scale": 1.0,
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


def parse_data(table, *, x_name, y_scale):
    """Return ``(label, x_values, y_values)`` per run, skipping ``MISSING`` cells.

    Times are multiplied by ``y_scale`` so every dataset lands on the shared axis.
    """
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
            y_values[name].append(float(value.replace(",", "")) * y_scale)
    return [
        (name.replace("_", " "), x_values[name], y_values[name])
        for name in headers[1:]
    ]


def plot_time(table, *, x_name, xlabel, y_scale):
    """Build a time/input-length line plot and return it without saving it."""
    series = parse_data(table, x_name=x_name, y_scale=y_scale)
    with plot_style(wide=True):
        fig, ax = plt.subplots()
        # One line per configuration. The style carries no linestyle, so each
        # run draws as a solid line with its own marker.
        plot_series(ax, series, styles=CONFIG_SERIES_STYLES)
        # Times land on whole milliseconds in both datasets, so integer ticks
        # read cleanly without a decimal point.
        format_axes(ax, xlabel=xlabel, ylabel=YLABEL, yformat="{x:,.0f}")
        finish_plot(ax, legend_outside=True)
    return fig, ax


def main():
    # Save inside the style context so the configured ``savefig.bbox`` (tight)
    # and font settings apply; otherwise figures clip at the canvas edges.
    with plot_style():
        for filename, dataset in datasets.items():
            fig, _ = plot_time(
                dataset["table"], x_name=dataset["x_name"],
                xlabel=dataset["xlabel"], y_scale=dataset["y_scale"],
            )
            fig.savefig(output_dir / filename, format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
