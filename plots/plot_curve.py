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
11047.0	1760.8	11046.0	1761.0	11046.0	1765.0
10387.7	1801.0	10757.1	1771.5	10738.0	1772.6
9731.5	1831.4	10279.6	1767.8	10247.0	1784.8
9075.2	1874.7	9759.2	1771.9	9727.0	1784.2
8414.0	1911.3	9226.3	1773.2	9185.0	1787.0
7757.7	1950.0	8677.1	1777.2	8634.0	1795.2
7101.5	1988.3	8122.2	1779.8	8073.0	1796.9
6971.4	2019.9	7556.7	1784.8	7507.0	1799.1
6928.5	2066.3	7024.4	1802.7	6972.0	1800.1"""
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
