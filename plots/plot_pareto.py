"""Plot the VRAM/time tradeoff of the three methods as a Pareto-style scatter.

The benchmark table lists, for each checkpoint layer, the VRAM footprint and
average time of every method. Lower is better on both axes, so the figure
scatters each method's configurations to expose the Pareto tradeoff.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import finish_plot, format_axes, plot_series, plot_style


data = """Checkpoint		Sparse		Sparse 15-bit
layers	vram	avg_time	vram	avg_time	vram	avg_time
0	12346.5	1790.8	12346.5	1790.6	12345.5	1790.4
1	11689.2	1836.9	12057.6	1794.0	12039.6	1795.4
2	11033.0	1885.9	11626.9	1792.3	11595.7	1798.6
3	10377.7	1926.7	11157.2	1794.5	11114.8	1801.9
4	9721.5	1971.8	10665.9	1794.9	10616.8	1804.3
5	9065.2	2016.8	10159.3	1795.6	10101.4	1805.7
6	8409.0	2061.9	9639.8	1796.3	9578.0	1807.3
7	7990.9	2107.5	9112.4	1796.7	9044.1	1808.3
8	8241.0	2153.1	7268.6	1797.6	7200.4	1809.7"""

save_filename = Path(__file__).resolve().with_name("relu2_80_sparse_pareto.pdf")


def parse_data(table):
    """Convert the two-header table's VRAM/time pairs into (label, x, y)."""
    lines = [line.strip() for line in table.strip().splitlines() if line.strip()]
    labels = [label.strip() for label in lines[0].split("\t") if label.strip()]
    rows = [[float(value) for value in line.split()] for line in lines[2:]]

    return [
        (
            label,
            [row[index * 2 + 2] for row in rows],
            [row[index * 2 + 1] for row in rows],
        )
        for index, label in enumerate(labels)
    ]


def plot_pareto(table):
    """Build the VRAM/time scatter figure and return it without saving it."""
    with plot_style():
        fig, ax = plt.subplots()
        plot_series(ax, parse_data(table), linestyle="none")
        format_axes(
            ax, xlabel="Average time (ms)", ylabel="VRAM (MiB)", yformat="{x:,.0f}",
        )
        finish_plot(ax)
    return fig, ax


def main():
    with plot_style():
        fig, _ = plot_pareto(data)
        fig.savefig(save_filename, format="pdf")
        plt.show()


if __name__ == "__main__":
    main()
