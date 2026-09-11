"""Reusable Matplotlib styling and axes-based plotting components."""

import colorsys
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#262626",
    "axes.labelcolor": "#262626",
    "axes.labelsize": 13,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.xmargin": 0.025,
    "axes.ymargin": 0.06,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
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

WIDE_FIGURE_SIZE = (8, 4.5)
LINE_STYLES = ("-", "--", "-.", ":")
NEUTRAL_COLOR = "#333333"
LAYOUT_PAD = 0.5

# Ordered groups (e.g. network layers) walk the hue wheel from red to violet,
# so the color advances with the group's index. The walk deliberately stops
# short of the full circle: wrapping 360 degrees puts the last group only
# 1/count back from the first, which makes layer 0 and layer 11 neighbours on
# the wheel so they read as the same red. Ending at three quarters of the wheel
# instead puts them at opposite ends of the spectrum (red vs violet) while still
# covering the whole rainbow. Lightness cycles through GROUP_CONTRAST_LEVELS
# independently of the hue, so neighbouring layers differ in tone as well as
# hue. Each tone's lightness is solved to hold its target contrast against
# white; 3.0 is the WCAG minimum for graphical objects.
GROUP_SATURATION = 0.85
GROUP_CONTRAST_LEVELS = (3.0, 4.6, 6.6)
GROUP_HUE_SPAN = 0.75


def _relative_luminance(rgb):
    """Return the WCAG relative luminance of an ``(r, g, b)`` triple in 0-1."""
    def linearize(channel):
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _color_at_hue(hue, saturation, target_luminance):
    """Return the ``(r, g, b)`` at ``hue`` whose luminance is ``target_luminance``.

    Lightness is found by bisection. Hues that cannot reach the target even at
    the lightness ceiling (saturated blues and reds) are returned at the ceiling
    so they stay as rich as possible rather than turning pale.
    """
    low, high = 0.0, 0.5
    for _ in range(30):
        middle = (low + high) / 2
        if _relative_luminance(colorsys.hls_to_rgb(hue, middle, saturation)) > target_luminance:
            high = middle
        else:
            low = middle
    return colorsys.hls_to_rgb(hue, (low + high) / 2, saturation)


def plot_style(overrides=None, *, wide=False):
    """Return a style context; use around figure creation, plotting and saving.

    ``overrides`` accepts Matplotlib rcParams. Previous settings are restored
    when the context exits, and importing this module has no styling effects.
    """
    params = {**PLOT_PARAMS}
    if wide:
        params["figure.figsize"] = WIDE_FIGURE_SIZE
    return plt.rc_context({**params, **(overrides or {})})


def plot_series(ax, series, *, styles=SERIES_STYLES, **line_kwargs):
    """Plot an iterable of ``(label, x_values, y_values)`` on an existing axes.

    Styles cycle across series. Matplotlib line keyword arguments override
    the shared styles. Return the created lines for further customization;
    callers control labels, legends, layout, saving and display.
    """
    styles = tuple(styles)
    if not styles:
        raise ValueError("styles must contain at least one line style")

    lines = []
    for (label, x_values, y_values), style in zip(series, cycle(styles)):
        options = {"markerfacecolor": "white", **style, **line_kwargs}
        lines.extend(ax.plot(x_values, y_values, label=label, **options))
    return lines


def sample_group_colors(count, *, saturation=GROUP_SATURATION,
                        contrast_levels=GROUP_CONTRAST_LEVELS,
                        hue_span=GROUP_HUE_SPAN):
    """Return ``count`` vivid colors ordered by the group's position.

    Hue walks from red to violet across ``hue_span`` of the wheel, so ordered
    groups (layer 0, layer 1, ...) read as a rainbow and the first and last
    groups land at opposite ends of it rather than next to each other. Lightness
    cycles through ``contrast_levels`` independently of the hue, which keeps
    neighbouring layers apart. Each tone's lightness is solved to hold its
    target contrast against white, so every color stays legible on white.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    levels = tuple(contrast_levels)
    if not levels:
        raise ValueError("contrast_levels must contain at least one value")
    if count == 1:
        offsets = [0.5]
    else:
        offsets = [index / (count - 1) for index in range(count)]
    return [
        _color_at_hue(
            hue_span * offset, saturation, 1.05 / levels[index % len(levels)] - 0.05,
        )
        for index, offset in enumerate(offsets)
    ]


def plot_grouped_series(ax, x, groups, metrics, *, colors=None):
    """Plot ``{group_label: {metric: y}}`` using color per group and style per metric.

    ``metrics`` is an ordered iterable of metric names: solid first, dashed
    second. Each group must supply every metric. Groups take the given
    ``colors`` in order, defaulting to the hue sweep from
    :func:`sample_group_colors` so the color reflects the group's position.
    Return group and metric legend handles for use with ``finish_plot``.
    Data parsing stays with the caller.
    """
    metric_styles = list(zip(metrics, cycle(LINE_STYLES)))
    items = list(groups.items())
    colors = sample_group_colors(len(items)) if colors is None else list(colors)
    if len(colors) != len(items):
        raise ValueError(f"expected {len(items)} colors, got {len(colors)}")

    group_handles = []
    for (label, values), color in zip(items, colors):
        for metric, linestyle in metric_styles:
            plot_series(
                ax, [(f"{label} {metric}", x, values[metric])],
                styles=[{"color": color, "linestyle": linestyle}],
            )
        group_handles.append(Line2D([], [], color=color, label=label))
    metric_handles = [
        Line2D([], [], color=NEUTRAL_COLOR, linestyle=style, label=metric)
        for metric, style in metric_styles
    ]
    return group_handles, metric_handles


def format_axes(ax, *, xlabel, ylabel, xformat="{x:,.0f}", yformat=None):
    """Apply axis labels and optional Matplotlib number-format strings."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    for axis, pattern in ((ax.xaxis, xformat), (ax.yaxis, yformat)):
        if pattern is not None:
            axis.set_major_formatter(StrMethodFormatter(pattern))


def finish_plot(ax, *, group_handles=None, metric_handles=None, group_title=None):
    """Apply shared legend placement and layout to an axes' figure.

    Ordinary series use an inside legend. Grouped series use a vertical legend
    and an optional metric key in reserved space on the right; pair this layout
    with ``plot_style(wide=True)``.
    """
    if group_handles is None:
        ax.legend(loc="best", borderaxespad=1.0)
        ax.figure.tight_layout(pad=LAYOUT_PAD)
        return

    legend = ax.legend(
        handles=group_handles, title=group_title, loc="upper left",
        bbox_to_anchor=(1.02, 1), ncol=1, borderaxespad=0,
    )
    if metric_handles is not None:
        ax.add_artist(legend)
        ax.legend(
            handles=metric_handles, loc="lower left",
            bbox_to_anchor=(1.02, 0), borderaxespad=0,
        )
    ax.figure.tight_layout(pad=LAYOUT_PAD)
