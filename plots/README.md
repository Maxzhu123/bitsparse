# Paper plots

Install Matplotlib and run the dataset file:

```sh
python3 -m pip install -r plots/requirements.in
python3 plots/plot_curve.py
```

Paste new data into `plot_curve.py` using the same two-header-row,
tab-separated format, and set `save_filename` to the desired PDF path. Shared
ICLR figure settings and plotting code live in `plot_lib.py`.
