"""Plot the VRAM/time tradeoff of the compressed methods as a Pareto scatter.

Each dataset is a benchmark table listing, for every checkpoint layer, the VRAM
footprint and average time of each method. Lower is better on both axes, so
plotting the two against each other exposes the tradeoff the methods trace out
as more layers are compressed, and the front can be read off directly.

The table groups each method's two metrics together, keeping one method's
measurements adjacent. Its first column is the checkpoint ordinal rather than an
axis, so the x values the parser reads go unused.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import finish_plot, format_axes, plot_series
from plot_tables import MIB_PER_GIB, parse_grouped, render


output_dir = Path(__file__).resolve().parent

# The first table column is the checkpoint ordinal, not an axis: the figure
# plots the two metrics against each other.
X_NAME = "layers"
METRICS = ("vram", "avg_time")

# Both axes carry the same quantity in every dataset, so the titles are shared
# rather than repeated per entry.
XLABEL = "Average time / ms"
YLABEL = "VRAM / GiB"

# Each table records VRAM in MiB; the axis is labelled in GiB.
datasets = {
    "relu2_90_pareto.pdf": """Checkpoint                  BitSparse                   Sign-bit
layers  vram     avg_time     vram     avg_time     vram     avg_time
0       11343.5  1830.7       11343.5  1830.7       11343.5  1830.7
1       10688.2  1870.6       11056.6  1829.4       11035.6  1832.6
2       10032.9  1909.5       10498.0  1827.8       10474.4  1832.9
3       9376.6   1949.8       9930.6   1825.5       9904.1   1833.4
4       8722.3   1988.5       9357.6   1824.0       9328.4   1832.1
5       8066.9   2029.4       8779.8   1821.9       8748.3   1832.4
6       7411.6   2068.7       8198.6   1820.4       8165.0   1830.6
7       6756.3   2108.8       7614.3   1818.8       7578.8   1830.3
8       6713.3   2148.2       5895.8   1815.9       5860.0   1829.2""",
    "relu_90_pareto.pdf": """Checkpoint                  BitSparse                   Sign-bit
layers  vram     avg_time     vram     avg_time     vram     avg_time
0       10039.0  1764.1       10039.0  1764.3       10039.0  1766.7
1       9382.7   1803.2       9749.1   1769.0       9731.1   1771.1
2       8726.4   1838.9       9155.1   1772.4       9131.8   1776.5
3       8070.1   1875.6       8557.9   1775.2       8533.5   1779.8
4       7413.8   1913.3       7958.9   1776.2       7933.4   1783.6
5       6757.4   1951.6       7354.0   1779.5       7327.8   1788.4
6       6101.1   1987.7       6752.0   1782.5       6724.9   1791.5
7       5933.1   2024.7       6148.8   1785.2       6121.0   1794.8
8       6057.1   2062.5       5741.5   1787.3       5714.7   1797.9""",
    "relu_50_pareto.pdf": """Checkpoint                  BitSparse                   Sign-bit
layers  vram     avg_time     vram     avg_time     vram     avg_time
0       10038.0  1770.8       10038.0  1767.2       10038.0  1767.5
1       9382.7   1807.2       9749.2   1771.7       9731.1   1775.0
2       8726.4   1844.1       9489.5   1774.3       9450.0   1780.8
3       8070.1   1882.4       9172.9   1778.6       9111.0   1786.6
4       7413.8   1918.2       8993.4   1780.9       8906.2   1794.3
5       6757.4   1955.6       8610.9   1784.0       8507.3   1799.1
6       6101.1   1993.7       8220.6   1787.8       8101.1   1802.9
7       5933.1   2030.4       7887.9   1790.7       7751.6   1807.5
8       6057.1   2066.9       7796.5   1793.4       7629.7   1817.6""",
    "relu2_50_pareto.pdf": """Checkpoint                  BitSparse                   Sign-bit
layers  vram     avg_time     vram     avg_time     vram     avg_time
0       11342.5  1835.8       11342.5  1831.3       11342.5  1831.2
1       10688.2  1873.2       11056.7  1831.2       11036.6  1832.9
2       10032.9  1912.6       10791.4  1829.9       10749.7  1835.8
3       9376.6   1953.4       10480.2  1830.1       10418.5  1843.0
4       8722.3   1996.5       10282.2  1830.9       10194.4  1842.4
5       8066.9   2035.6       9923.0   1828.4       9819.3   1842.9
6       7411.6   2075.4       9554.1   1827.0       9434.7   1847.3
7       6756.3   2113.0       9230.0   1826.0       9093.7   1845.6
8       6713.3   2153.4       7803.8   1824.5       7634.0   1846.7""",
}


def plot_pareto(table):
    """Build one VRAM/time figure and return it without saving it."""
    _, groups = parse_grouped(table, x_name=X_NAME, metrics=METRICS)
    # Convert the benchmark's MiB onto the GiB axis.
    series = [
        (label, columns["avg_time"], [vram / MIB_PER_GIB for vram in columns["vram"]])
        for label, columns in groups.items()
    ]

    fig, ax = plt.subplots()
    # Join each method's checkpoints in layer order so the trajectory as more
    # layers are compressed stays visible. Thin solid lines with the white-faced
    # markers drawn on top of them.
    plot_series(
        ax, series, linestyle="-", linewidth=1.0,
        markersize=5.0, markeredgewidth=1.1,
    )
    format_axes(ax, xlabel=XLABEL, ylabel=YLABEL, yformat="{x:,.1f}")
    finish_plot(ax)
    return fig, ax


def main():
    render(datasets, plot_pareto, output_dir=output_dir)


if __name__ == "__main__":
    main()
