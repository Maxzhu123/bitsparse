from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


PLOT_PARAMS = {
    "figure.figsize": (5.5, 3.4),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "font.size": 9,
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#262626",
    "axes.labelcolor": "#262626",
    "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SERIES_STYLES = (
    {"color": "#333333", "marker": "o", "linestyle": "-"},
    {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    {"color": "#D55E00", "marker": "^", "linestyle": "-."},
)

plt.rcParams.update(PLOT_PARAMS)


def _parse_data(table):
    lines = [line.strip() for line in table.strip().splitlines() if line.strip()]
    labels = [label.strip() for label in lines[0].split("\t") if label.strip()]
    rows = [[float(value) for value in line.split()] for line in lines[2:]]

    return [
        (
            label,
            [row[index * 2 + 1] for row in rows],
            [row[index * 2] for row in rows],
        )
        for index, label in enumerate(labels)
    ]


def plot_data(table, save_filename):
    fig, ax = plt.subplots()

    for (label, avg_times, vram_values), style in zip(
        _parse_data(table), cycle(SERIES_STYLES)
    ):
        ax.plot(
            avg_times,
            vram_values,
            label=label,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            **style,
        )

    ax.set_xlabel("Average time (ms)")
    ax.set_ylabel("VRAM (MiB)")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.margins(x=0.025, y=0.06)
    ax.legend(loc="best", borderaxespad=1.0)
    fig.tight_layout(pad=0.35)
    fig.savefig(save_filename, format="pdf")
    plt.show()
