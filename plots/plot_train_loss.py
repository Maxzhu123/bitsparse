"""Validation loss over a full nanoGPT training run.

The supplied measurements are validation losses, as extracted by
``nanogpt/parse_logs.py``. Normal training uses the dense baseline's appearance;
BitSparse-sb uses its shared colour and marker with a dashed line so both runs
remain visible where their losses nearly coincide.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import (
    CONFIG_STYLES, WIDE_FONT_SCALE, finish_plot, format_axes, plot_series,
)
from plot_tables import parse_series, render


output_dir = Path(__file__).resolve().parent

datasets = {
    "nanogpt_train_loss.pdf": """
Step Normal BitSparse-sb
0 10.82587 10.82587
125 4.67375 4.65882
250 4.12124 4.12057
375 3.93842 3.9371
500 3.83492 3.83332
625 3.76411 3.76363
750 3.71702 3.718
875 3.67836 3.67557
1000 3.64128 3.64147
1125 3.61302 3.61528
1250 3.582 3.58117
1375 3.55553 3.555
1500 3.52603 3.52849
1625 3.50695 3.50777
1750 3.48532 3.48533
1875 3.4654 3.46459
2000 3.44609 3.44499
2125 3.42765 3.42776
2250 3.40947 3.40975
2375 3.39328 3.39357
2500 3.37681 3.3767
2625 3.35926 3.35993
2750 3.34284 3.3435
2875 3.32699 3.32718
3000 3.31107 3.31122
3025 3.30712 3.30731
3050 3.30407 3.30436
3075 3.30114 3.30136
3100 3.29833 3.2986
3125 3.29561 3.29586
3150 3.29247 3.2928
3175 3.2899 3.29016
3200 3.28744 3.28759
3225 3.28507 3.28527
3250 3.28298 3.28317
3275 3.28117 3.28131
3300 3.27958 3.27975
3325 3.27841 3.27858
3350 3.27782 3.278
""",
}


def plot_train_loss(table):
    """Build the validation-loss comparison and return it without saving it."""
    series = parse_series(table, x_name="Step")
    styles = {
        "Normal": {**CONFIG_STYLES["Dense"], "markevery": (0, 4)},
        "BitSparse-sb": {
            **CONFIG_STYLES["BitSparse-sb"],
            "linestyle": "--",
            "markevery": (2, 4),
        },
    }

    fig, ax = plt.subplots()
    # Stagger only the markers; both lines include every supplied checkpoint.
    plot_series(ax, series, styles=styles)
    format_axes(ax, xlabel="Training step", ylabel="Validation loss", x_step=1000)
    finish_plot(ax)

    # Zoom from step 3000 through the end of training.
    # Reuse the same series, with a marker at every checkpoint in the inset.
    zoom = ax.inset_axes([0.48, 0.32, 0.48, 0.36])
    plot_series(ax=zoom, series=series, styles=styles, markevery=1)
    zoom.set(xlim=(3000, 3350), ylim=(3.276, 3.313))
    format_axes(
        zoom, xlabel="", ylabel="", yformat="{x:.2f}",
    )
    zoom.set_xticks([3000, 3125, 3250, 3350])
    zoom.set_yticks([3.28, 3.29, 3.30, 3.31])
    zoom.tick_params(labelsize=11, pad=4)
    zoom.set_title("Final training steps", fontsize=12)
    zoom.spines[["top", "right"]].set_visible(True)
    ax.indicate_inset_zoom(zoom, edgecolor=CONFIG_STYLES["Dense"]["color"])
    return fig, ax


def main():
    render(
        datasets, plot_train_loss, output_dir=output_dir,
        wide=True, font_scale=WIDE_FONT_SCALE,
    )


if __name__ == "__main__":
    main()
