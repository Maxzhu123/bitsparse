"""Table parsing and figure rendering shared by the figure scripts.

Every figure script keeps its measurements in a text table so the numbers stay
visible in the diff. This module turns those tables into the series tuples the
plotting helpers consume, and owns the render loop so the four scripts do not
each reimplement saving.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import plot_style


# Written where a run was not measured at that x value. Gaps have to be spelled
# out: splitting on whitespace collapses an empty cell, which would silently
# shift the following values into the wrong series.
MISSING = "-"

# The benchmark harnesses report VRAM in MiB; the figures use GiB.
MIB_PER_GIB = 1024.0


def _split_rows(table):
    """Return the table's rows as token lists, dropping blank lines."""
    return [line.split() for line in table.splitlines() if line.strip()]


def _headers(rows, x_name, expected_columns=None):
    """Validate the row/column shape and return the header row.

    ``expected_columns`` is checked when the caller knows how wide every row
    must be, which catches a row whose gaps were not spelled out.
    """
    if len(rows) < 2 or rows[0][0] != x_name:
        raise ValueError(f"expected a {x_name!r} header and at least one data row")

    headers = rows[0]
    if len(set(headers)) != len(headers):
        raise ValueError(f"duplicate column names in {headers}")

    expected = expected_columns if expected_columns is not None else len(headers)
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected:
            raise ValueError(
                f"row {row_number}: expected {expected} values, got {len(row)}"
            )
    return headers


def _value(token):
    """Parse one cell, allowing thousands separators."""
    return float(token.replace(",", ""))


def parse_series(table, *, x_name, y_scale=1.0):
    """Parse a table of ``x`` plus one column per series.

    The first line names the x column and each series; every later line is one
    x value followed by that series' measurement, with ``MISSING`` for gaps.
    Measurements are multiplied by ``y_scale`` so datasets recorded in different
    units can share an axis. Return ``(label, x_values, y_values)`` per series,
    in column order.
    """
    rows = _split_rows(table)
    headers = _headers(rows, x_name)

    x_values = {name: [] for name in headers[1:]}
    y_values = {name: [] for name in headers[1:]}
    for row in rows[1:]:
        x = _value(row[0])
        for name, token in zip(headers[1:], row[1:]):
            if token == MISSING:
                continue
            x_values[name].append(x)
            y_values[name].append(_value(token) * y_scale)

    return [(name, x_values[name], y_values[name]) for name in headers[1:]]


def parse_grouped(table, *, x_name, metrics):
    """Parse a table holding several metrics per series.

    The first line names each series; the second names the columns, starting
    with ``x_name`` and then one entry per metric per series, so series ``i``
    owns the metrics in order. Return ``{label: {metric: (x, values)}}``.

    Unlike :func:`parse_series` the columns are grouped rather than flat, which
    keeps each series' measurements adjacent and readable for a tradeoff plot.
    """
    rows = _split_rows(table)
    headers = _headers(rows, x_name)

    labels = rows[0][1:]
    metric_names = list(metrics)
    expected = 1 + len(labels) * len(metric_names)
    if len(headers) != expected:
        raise ValueError(
            f"expected {expected} columns for {len(labels)} series x "
            f"{len(metric_names)} metrics, got {len(headers)}"
        )

    series = {}
    for index, label in enumerate(labels):
        columns = {}
        for offset, metric in enumerate(metric_names):
            column = 1 + index * len(metric_names) + offset
            columns[metric] = ([], [])
            for row in rows[1:]:
                token = row[column]
                if token == MISSING:
                    continue
                columns[metric][0].append(_value(row[0]))
                columns[metric][1].append(_value(token))
        series[label] = columns
    return series


def render(datasets, build, *, output_dir, font_scale=1.0):
    """Save one PDF per dataset and show the figures.

    ``datasets`` maps an output filename to whatever ``build`` accepts. Both the
    save and the show run inside one style context: ``plt.show`` redraws, and a
    redraw under the default rcParams would size the text for a different font
    than :func:`~plot_lib.plot_style` selected.
    """
    with plot_style(font_scale=font_scale):
        for filename, dataset in datasets.items():
            fig, _ = build(dataset)
            fig.savefig(Path(output_dir) / filename, format="pdf")
        plt.show()

