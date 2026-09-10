"""Plot each layer's average and least-sparse values during training."""

from pathlib import Path
import re

import matplotlib.pyplot as plt

from plot_lib import finish_plot, format_axes, plot_grouped_series, plot_style


data = """
step	layer_0_avg_sparsity	layer_1_avg_sparsity	layer_2_avg_sparsity	layer_3_avg_sparsity	layer_4_avg_sparsity	layer_5_avg_sparsity	layer_6_avg_sparsity	layer_7_avg_sparsity	layer_8_avg_sparsity	layer_9_avg_sparsity	layer_10_avg_sparsity	layer_11_avg_sparsity	layer_0_least_sparse	layer_1_least_sparse	layer_2_least_sparse	layer_3_least_sparse	layer_4_least_sparse	layer_5_least_sparse	layer_6_least_sparse	layer_7_least_sparse	layer_8_least_sparse	layer_9_least_sparse	layer_10_least_sparse	layer_11_least_sparse
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
3300	0.308	0.101	0.108	0.135	0.136	0.140	0.14770974218845367	0.148	0.162	0.176	0.210	0.344	0.406	0.106	0.114	0.141	0.140	0.145	0.153	0.153	0.170	0.188	0.222	0.370
"""

save_filename = Path(__file__).resolve().with_name("train_sparsity.pdf")


def parse_data(table):
    """Read the average and least-sparse metric columns for each layer."""
    rows = [line.split() for line in table.splitlines() if line.strip()]
    if len(rows) < 2 or rows[0][0] != "step":
        raise ValueError("Expected a step header and at least one data row")
    headers = rows[0]
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate column names")
    values = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(headers):
            raise ValueError(f"Row {row_number}: expected {len(headers)} values, got {len(row)}")
        values.append([float(value) for value in row])

    layers = {}
    for index, header in enumerate(headers[1:], start=1):
        match = re.fullmatch(r"layer_(\d+)_(avg_sparsity|least_sparse)", header)
        if match is None:
            raise ValueError(f"Unrecognized metric column: {header}")
        layer, metric = match.groups()
        layers.setdefault(int(layer), {})[metric] = [row[index] for row in values]
    return [row[0] for row in values], dict(sorted(layers.items()))


def plot_train_sparsity(table):
    """Build a line plot, pairing each layer's metrics by color."""
    steps, layers = parse_data(table)
    with plot_style(wide=True):
        fig, ax = plt.subplots()
        layer_handles, _ = plot_grouped_series(
            ax, steps,
            {str(layer): metrics for layer, metrics in layers.items()},
            ("avg_sparsity", "least_sparse"),
        )
        format_axes(ax, xlabel="Training step", ylabel="Sparsity")
        finish_plot(
            ax, group_handles=layer_handles,
            group_title="Layer",
        )
    return fig, ax


def main():
    with plot_style():
        fig, _ = plot_train_sparsity(data)
        fig.savefig(save_filename, format="pdf")
        plt.show()


if __name__ == "__main__":
    main()
