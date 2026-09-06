# `lib_sparse`

`lib_sparse` is a CUDA/Triton library for compressing the positive activations of ReLU and ReLU² feed-forward networks. It stores a tile-wise bitmask, a compact stream of active values, and a prefix index for locating each tile's values. The compressed activation can then be retained by autograd instead of retaining a full dense activation tensor.

The project is designed around PyTorch and custom autograd functions. It supports BF16 storage and FP8 storage/arithmetic, and it can optionally remove the sign bit from every non-negative value and pack the remaining bits into a continuous bitstream.

## Read the documentation

- [Usage guide](usage.md): installation assumptions, compression/decompression, matrix products, FFN layers, FP8, and shared buffers.
- [Architecture](architecture.md): the representation, kernels, forward/backward data flow, and integration with the Nemotron model.
- [Performance notes](performance.md): memory layout, asynchronous tricks, autotuning, in-place operations, and trade-offs.

## The short version

For a dense ReLU activation `h` with shape `[M, N]`:

1. The matrix is divided into 64×64 tiles.
2. Each tile gets a 1-bit-per-element mask for `h > 0` and a count of active values.
3. A prefix sum converts tile counts into offsets in one global value stream.
4. Positive values are scattered into that stream in tile-major order.
5. The resulting `BitsparseTensor` keeps the mask, value stream, prefix array, shape, dtype, and optional FP8 scale.
6. During backward, Triton kernels use the saved mask/value stream to reconstruct or apply the activation derivative.

The default high-level FFN path still computes its forward matmuls densely. `AspB`, `AspRelu2B`, and `spAB` unpack a sparse operand to dense before using a dense/FP8 matmul. The main benefit is therefore lower activation-storage VRAM, with compression and decompression overhead, rather than skipping multiply-adds for zero entries.

## Source map

| Area | Implementation |
| --- | --- |
| Sparse metadata and shared storage | [`lib_sparse/bitsparse.py`](../lib_sparse/bitsparse.py) |
| Tile compression entry point | [`lib_sparse/src/functions.py`](../lib_sparse/src/functions.py) |
| Triton launch wrappers | [`lib_sparse/src/triton_operators.py`](../lib_sparse/src/triton_operators.py) |
| Tile, unpack, and gradient kernels | [`lib_sparse/src/triton_kernels.py`](../lib_sparse/src/triton_kernels.py) |
| Continuous 7/15-bit packing | [`lib_sparse/src/bitpacking.py`](../lib_sparse/src/bitpacking.py) |
| Sparse matmul helpers | [`lib_sparse/src/sparse_matmul.py`](../lib_sparse/src/sparse_matmul.py) |
| ReLU/ReLU² autograd FFNs | [`lib_sparse/layers.py`](../lib_sparse/layers.py) |
| FP8 conversion and `scaled_mm` wrapper | [`lib_sparse/fp8.py`](../lib_sparse/fp8.py) |
| Tile and packing constants | [`lib_sparse/config.py`](../lib_sparse/config.py) |

The package-level `lib_sparse/__init__.py` is empty. Import symbols from their defining modules, as shown in the usage guide.

