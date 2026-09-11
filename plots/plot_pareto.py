"""Plot the VRAM/time tradeoff of the three methods as a Pareto-style scatter.

Each entry in ``datasets`` maps an output filename to a benchmark table listing,
for every checkpoint layer, the VRAM footprint and average time of each method.
Lower is better on both axes, so each figure scatters the methods' configurations
to expose the Pareto tradeoff. ``main`` renders one PDF per dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import finish_plot, format_axes, plot_series, plot_style


output_dir = Path(__file__).resolve().parent

MIB_PER_GIB = 1024.0


datasets = {
    "relu2_80_pareto.pdf": """Checkpoint		Sparse		Sparse 15-bit
layers	vram	avg_time	vram	avg_time	vram	avg_time
0	12346.5	1790.8	12346.5	1790.6	12345.5	1790.4
1	11689.2	1836.9	12057.6	1794.0	12039.6	1795.4
2	11033.0	1885.9	11626.9	1792.3	11595.7	1798.6
3	10377.7	1926.7	11157.2	1794.5	11114.8	1801.9
4	9721.5	1971.8	10665.9	1794.9	10616.8	1804.3
5	9065.2	2016.8	10159.3	1795.6	10101.4	1805.7
6	8409.0	2061.9	9639.8	1796.3	9578.0	1807.3
7	7990.9	2107.5	9112.4	1796.7	9044.1	1808.3
8	8241.0	2153.1	7268.6	1797.6	7200.4	1809.7""",
    "relu_80_pareto.pdf": """Checkpoint		Sparse		Sparse 15-bit
layers	vram	avg_time	vram	avg_time	vram	avg_time
0	11047.0	1747.39	11046.0	1744.6	11046.0	1744.65
1	10387.7	1783.74	10757.1	1747.64	10738.0	1750.94
2	9731.5	1818.25	10279.6	1751.01	10247.0	1755.61
3	9075.2	1857.62	9759.2	1754.31	9727.0	1759.61
4	8414.0	1898.96	9226.3	1769.53	9185.0	1777.29
5	7757.7	1942.43	8677.1	1768.47	8634.0	1774.97
6	7101.5	1983.73	8122.2	1776.86	8073.0	1785.76
7	6971.4	2016.24	7556.7	1774.31	7507.0	1782.76
8	6928.5	2053.16	7024.4	1776.74	6972.0	1786.1""",
    "relu_50_pareto.pdf": """Checkpoint		Sparse		Sparse 15-bit
layers	vram	avg_time	vram	avg_time	vram	avg_time
0	11047.0	1749.29	11047.0	1745.96	11047.0	1745.94
1	10387.7	1782.58	10758.2	1748.42	10740.1	1758.32
2	9731.5	1824.7	10497.5	1755.94	10460.0	1761.44
3	9075.2	1862.3	10175.9	1759.66	10117.0	1767.35
4	8414.0	1900.66	9999.6	1762.91	9915.2	1774.8
5	7757.7	1938.73	9616.1	1765.75	9511.3	1780.27
6	7101.5	1977.2	9221.9	1768.94	9109.1	1783.38
7	6971.4	2013.15	8889.2	1772.31	8755.5	1790.2
8	6928.5	2050.97	8799.7	1783.74	8632.2	1804.72""",
    "relu2_50_pareto.pdf": """Checkpoint		Sparse		Sparse 15-bit
layers	vram	avg_time	vram	avg_time	vram	avg_time
0	12346.5	1807.8	12346.5	1803.6	12346.5	1803.4
1	11689.2	1852.4	12057.7	1804.5	12057.7	1807.2
2	11033.0	1895.8	11792.4	1806.1	11792.4	1812.0
3	10377.7	1942.8	11482.2	1808.0	11482.2	1816.6
4	9721.5	1988.5	11285.2	1808.6	11285.2	1821.1
5	9065.2	2033.8	10926.0	1809.9	10926.0	1824.9
6	8409.0	2078.1	10554.7	1811.3	10554.7	1828.0
7	7990.9	2121.3	10232.0	1814.1	10232.0	1831.2
8	8241.0	2166.8	8805.5	1814.8	8805.5	1837.3""",
}

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
    """Build the VRAM/time figure (VRAM in GiB) and return it without saving it."""
    # Benchmarks report VRAM in MiB; scale to GiB for a more readable axis.
    series = [
        (label, times, [vram / MIB_PER_GIB for vram in vrams])
        for label, times, vrams in parse_data(table)
    ]
    with plot_style():
        fig, ax = plt.subplots()
        # Join each method's checkpoints in layer order so the trajectory as
        # more layers are compressed stays visible. Thin solid lines in each
        # series' color, with the white-faced markers drawn on top of the line.
        plot_series(
            ax, series, linestyle="-", linewidth=1.0,
            markersize=5.0, markeredgewidth=1.1,
        )
        format_axes(
            ax, xlabel="Average time / ms", ylabel="VRAM / GiB", yformat="{x:,.1f}",
        )
        finish_plot(ax)
    return fig, ax


def main():
    # Save inside the style context so the configured ``savefig.bbox`` (tight)
    # and font settings apply; otherwise the figure is written at the raw
    # canvas edges and the rotated y-label gets clipped.
    with plot_style():
        for filename, table in datasets.items():
            fig, _ = plot_pareto(table)
            fig.savefig(output_dir / filename, format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
