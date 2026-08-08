# data = """Checkpoint		Sparse		Sparse 15bit
# vram	avg_time	vram	avg_time	vram	avg_time
# 11047.0	1760.8	11047.0	1756.7	11047.0	1759.0
# 10387.7	1801.0	10758.2	1758.5	10740.1	1765.5
# 9731.5	1831.4	10497.5	1764.0	10460.0	1771.3
# 9075.2	1874.7	10175.9	1772.6	10117.0	1777.0
# 8414.0	1911.3	9999.6	1775.3	9915.2	1783.5
# 7757.7	1950.0	9616.1	1787.7	9511.3	1790.2
# 7101.5	1988.3	9221.9	1792.1	9109.1	1794.3
# 6971.4	2019.9	8889.2	1794.9	8755.5	1798.8
# 6928.5	2066.3	8799.7	1799.5	8632.2	1807.0"""

data = """Checkpoint		Sparse		Sparse 15bit	
vram	avg_time	vram	avg_time	vram	avg_time
12346.5	1826.8	12346.5	1822.2	12346.5	1822.8
11689.2	1885.9	12057.7	1822.0	12057.7	1829.0
11033.0	1921.9	11792.4	1821.4	11792.4	1830.0
10377.7	1971.6	11482.2	1821.3	11482.2	1829.8
9721.5	2015.8	11285.2	1821.2	11285.2	1832.6
9065.2	2052.0	10926.0	1818.1	10926.0	1836.4
8409.0	2099.6	10554.7	1817.2	10554.7	1836.7
7990.9	2145.1	10232.0	1814.8	10232.0	1840.0
8241.0	2187.8	8805.5	1815.8	8805.5	1842.8"""
import matplotlib.pyplot as plt


def parse_data(table):
    """Return the three series as (label, avg_times, vram_values)."""
    lines = [line.strip() for line in table.strip().splitlines() if line.strip()]
    labels = [label.strip() for label in lines[0].split("\t") if label.strip()]
    rows = [[float(value) for value in line.split()] for line in lines[2:]]

    series = []
    for index, label in enumerate(labels):
        vram_column = index * 2
        time_column = vram_column + 1
        avg_times = [row[time_column] for row in rows]
        vram_values = [row[vram_column] for row in rows]
        series.append((label, avg_times, vram_values))

    return series


def plot_data(table):
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, avg_times, vram_values in parse_data(table):
        ax.plot(avg_times, vram_values, marker="o", linewidth=2, label=label)

    ax.set_xlabel("Average time")
    ax.set_ylabel("VRAM")
    ax.set_title("VRAM vs. Average Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    plot_data(data)
    plt.show()
