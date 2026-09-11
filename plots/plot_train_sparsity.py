"""Plot each layer's average and least-sparse density during training.

The table names one column per layer and metric, so the layers are read back out
of the column names and are coloured as an ordered series: the colour tracks
depth and the line style carries the metric.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import finish_plot, format_axes, plot_grouped_series
from plot_tables import parse_series, render


output_dir = Path(__file__).resolve().parent

X_NAME = "step"
METRICS = ("avg_sparsity", "least_sparse")
COLUMN = re.compile(r"layer_(\d+)_(avg_sparsity|least_sparse)")
YLABEL = r"$\rho$"

datasets = {
    "train_sparsity.pdf": """step	layer_0_avg_sparsity	layer_1_avg_sparsity	layer_2_avg_sparsity	layer_3_avg_sparsity	layer_4_avg_sparsity	layer_5_avg_sparsity	layer_6_avg_sparsity	layer_7_avg_sparsity	layer_8_avg_sparsity	layer_9_avg_sparsity	layer_10_avg_sparsity	layer_11_avg_sparsity	layer_0_least_sparse	layer_1_least_sparse	layer_2_least_sparse	layer_3_least_sparse	layer_4_least_sparse	layer_5_least_sparse	layer_6_least_sparse	layer_7_least_sparse	layer_8_least_sparse	layer_9_least_sparse	layer_10_least_sparse	layer_11_least_sparse
0	0.500	0.501	0.499	0.502	0.500	0.500	0.5004469156265259	0.503	0.501	0.500	0.501	0.499	0.500	0.502	0.500	0.502	0.500	0.501	0.501	0.503	0.501	0.500	0.501	0.500
300	0.340	0.113	0.111	0.112	0.113	0.117	0.12577097117900848	0.139	0.175	0.200	0.249	0.341	0.406	0.118	0.117	0.119	0.119	0.122	0.130	0.144	0.180	0.205	0.257	0.352
600	0.332	0.107	0.107	0.114	0.112	0.118	0.1277935951948166	0.136	0.161	0.179	0.224	0.335	0.411	0.111	0.112	0.120	0.118	0.123	0.132	0.140	0.168	0.187	0.234	0.350
900	0.330	0.106	0.105	0.119	0.113	0.119	0.12941671907901764	0.133	0.154	0.167	0.207	0.323	0.415	0.110	0.110	0.123	0.118	0.124	0.134	0.138	0.160	0.174	0.215	0.348
1200	0.331	0.099	0.103	0.117	0.112	0.116	0.1248483955860138	0.128	0.148	0.159	0.196	0.317	0.416	0.103	0.107	0.122	0.116	0.121	0.129	0.133	0.156	0.169	0.205	0.344
1500	0.330	0.100	0.100	0.118	0.114	0.118	0.1261550486087799	0.130	0.147	0.158	0.194	0.313	0.422	0.105	0.104	0.123	0.118	0.122	0.131	0.134	0.156	0.169	0.204	0.339
1800	0.326	0.097	0.099	0.117	0.115	0.119	0.1271994560956955	0.129	0.145	0.155	0.189	0.313	0.426	0.102	0.104	0.122	0.119	0.125	0.132	0.133	0.152	0.166	0.199	0.343
2100	0.324	0.097	0.099	0.120	0.118	0.122	0.12964564561843872	0.133	0.147	0.159	0.191	0.316	0.424	0.102	0.104	0.125	0.121	0.127	0.134	0.138	0.156	0.171	0.203	0.347
2400	0.320	0.097	0.101	0.123	0.123	0.125	0.13235673308372498	0.135	0.151	0.162	0.195	0.321	0.421	0.103	0.106	0.128	0.127	0.131	0.137	0.140	0.159	0.175	0.208	0.350
2700	0.317	0.099	0.104	0.127	0.127	0.130	0.13879603147506714	0.141	0.155	0.167	0.200	0.330	0.418	0.105	0.109	0.132	0.131	0.135	0.145	0.146	0.164	0.179	0.212	0.356
3000	0.311	0.101	0.106	0.131	0.132	0.135	0.14417055249214172	0.144	0.159	0.172	0.206	0.339	0.410	0.105	0.111	0.137	0.136	0.140	0.149	0.149	0.165	0.183	0.218	0.363
3300	0.308	0.101	0.108	0.135	0.136	0.140	0.14770974218845367	0.148	0.162	0.176	0.210	0.344	0.406	0.106	0.114	0.141	0.140	0.145	0.153	0.153	0.170	0.188	0.222	0.370""",
}


def group_layers(table):
    """Return ``(steps, {layer: {metric: values}})``.

    The columns arrive in one flat row, so each is matched against
    :data:`COLUMN` to recover the layer it belongs to and the metric it carries.
    """
    steps, layers = None, {}
    for name, x_values, y_values in parse_series(table, x_name=X_NAME):
        match = COLUMN.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected metric column: {name!r}")
        layer, metric = match.groups()
        steps = x_values
        layers.setdefault(int(layer), {})[metric] = y_values
    return steps, dict(sorted(layers.items()))


def plot_train_sparsity(table):
    """Build the training-sparsity figure and return it without saving it."""
    steps, layers = group_layers(table)

    fig, ax = plt.subplots()
    # The layers are ordered by depth, and plot_grouped_series colours each by
    # its position, so the ramp reads as depth without further work here.
    handles, _ = plot_grouped_series(
        ax, steps,
        {str(layer): metrics for layer, metrics in layers.items()},
        METRICS,
    )
    format_axes(ax, xlabel="Training step", ylabel=YLABEL)
    finish_plot(ax, group_handles=handles, group_title="Layer")
    return fig, ax


def main():
    render(datasets, plot_train_sparsity, output_dir=output_dir, wide=True)


if __name__ == "__main__":
    main()
