"""Plot the VRAM/time tradeoff from the benchmark table below."""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import finish_plot, format_axes, plot_series, plot_style


data = """Checkpoint		Sparse		Sparse 15-bit
vram	avg_time	vram	avg_time	vram	avg_time
11047.0	1760.8	11046.0	1761.0	11046.0	1765.0
10387.7	1801.0	10757.1	1771.5	10738.0	1772.6
9731.5	1831.4	10279.6	1767.8	10247.0	1784.8
9075.2	1874.7	9759.2	1771.9	9727.0	1784.2
8414.0	1911.3	9226.3	1773.2	9185.0	1787.0
7757.7	1950.0	8677.1	1777.2	8634.0	1795.2
7101.5	1988.3	8122.2	1779.8	8073.0	1796.9
6971.4	2019.9	7556.7	1784.8	7507.0	1799.1
6928.5	2066.3	7024.4	1802.7	6972.0	1800.1"""

save_filename = Path(__file__).resolve().with_name("vram_vs_time.pdf")


def parse_data(table):
    """Convert the two-header table's VRAM/time pairs into (label, x, y)."""
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


def plot_curve(table):
    """Build the benchmark figure and return it without saving or showing it."""
    with plot_style():
        fig, ax = plt.subplots()
        plot_series(ax, parse_data(table))
        format_axes(
            ax, xlabel="Average time (ms)", ylabel="VRAM (MiB)", yformat="{x:,.0f}",
        )
        finish_plot(ax)
    return fig, ax


def main():
    with plot_style():
        fig, _ = plot_curve(data)
        fig.savefig(save_filename, format="pdf")
        plt.show()


if __name__ == "__main__":
    main()
