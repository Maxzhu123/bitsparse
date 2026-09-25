# BitSparse

Code for **BitSparse: Compact Activation Storage for ReLU Feed-Forward Networks**.
BitSparse stores nonzero activations with a packed bitmask and tile offsets,
then reconstructs them during backward. It supports ReLU and squared ReLU,
BF16 and FP8 storage, optional sign-bit packing, and a shared value buffer.

## Setup

Use Linux with an NVIDIA GPU, a CUDA-enabled PyTorch installation, and Triton.
FP8 matrix multiplication requires a compatible GPU and PyTorch build. The
implementation uses recent PyTorch APIs, including `torch.nn.functional.scaled_mm`
and fused RMS normalization.

From the repository root, install the Python dependencies into your environment:

```bash
pip install -r requirements.txt
pip install -r nanogpt/data/requirements.txt
```

The nvCOMP baseline additionally requires NVIDIA's nvCOMP Python package
(`nvidia.nvcomp`). Nemotron requires Mamba and causal-convolution kernels compatible
with the installed PyTorch and CUDA versions; it also imports `mamba_ssm` directly.
These dependencies are not all included in `requirements.txt`.

## Repository layout

- `lib_sparse/`: compression kernels, sparse tensor storage, FP8 utilities, and autograd layers.
- `experiments/`: synthetic MLP benchmarks, compression baselines, validation, and ablations.
- `nanogpt/`: training, evaluation, and sparsity measurements for Modded-NanoGPT.
- `nemotron/`: Nemotron-H implementation and evaluation scripts.
- `results/`: output directory for benchmark CSV files.
- `program.md`: prompt used during kernel optimization.

## Validation and MLP benchmarks

Run these commands from the repository root:

```bash
python -m experiments.validate
python -m experiments.benchmark
python -m experiments.run_relu
```

The default validation checks BF16 compression and reconstruction with and without
sign-bit packing and a shared buffer. The MLP script compares dense storage,
checkpointing, BitSparse, and BitSparse with sign-bit packing.

Additional experiments:

```bash
python -m experiments.baseline_csr
python -m experiments.baseline_nvcomp
python -m experiments.baseline_compact
python -m experiments.ablation_encode_decode
python -m experiments.ablation_relu_backward
```

Settings are edited in the scripts rather than supplied as command-line arguments.
Check `DTYPE`, `DATA_SPARSITY`, and `BASIC_MODE` in `experiments/experiment.py`,
the configurations in each entry point, and `lib_sparse/config.py` before running.
The nvCOMP baseline has a separate `USE_FP8` setting. Large benchmark shapes may
need to be reduced to fit your GPU. Several scripts append results to existing CSVs.

## Language-model experiments

Place pretokenized FineWeb10B shards in `nanogpt/data/fineweb10B/`, using the
Modded-NanoGPT binary format. Expected filenames include `fineweb_train_*.bin`
and `fineweb_val_*.bin`. Dataset shards and model checkpoints are not included.

Run NanoGPT scripts from `nanogpt/`, with the repository root on the import path:

```bash
cd nanogpt
PYTHONPATH=.. python train_gpt_simple.py
PYTHONPATH=.. python test_time_mem.py
```

Training settings, including BitSparse, sign-bit packing, buffer capacity, and
microbatch size, are defined in the scripts. Training writes checkpoints and logs
under `nanogpt/logs/`. To run `eval_gpt.py` or `sparsity_analysis.py`, first update
their checkpoint paths and `LOG_DIR` to an available run and create the output
directory. Their current defaults refer to a local July training run.

For Nemotron sparsity measurements, run from the repository root:

```bash
python -m nemotron.sparsity_analysis
```

The script loads `nvidia/Nemotron-H-8B-Base-8K` and reads the FineWeb validation
shards. `FINEWEB_ROOT` can override the default data root, `nanogpt/data/`.
The separate timing script, `python -m nemotron.start`, currently requires
`nemotron/sample_text.txt`; supply that input or update its data-loading code
before running it.
