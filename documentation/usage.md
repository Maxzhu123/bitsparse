# Using `lib_sparse`

## Requirements and operating assumptions

The compression and unpacking paths launch Triton kernels and are intended for a CUDA GPU. The validation script explicitly requires CUDA. The environment also needs a PyTorch build with the relevant FP8 dtypes and `torch.nn.functional.scaled_mm` when FP8 matmul is used. The repository's [`requirements.txt`](../requirements.txt) lists the Python-level dependencies, but does not itself pin a CUDA, Triton, or PyTorch version.

Install the repository in a way that leaves its root on `PYTHONPATH`, then import the modules directly:

```python
import torch

from lib_sparse.bitsparse import TensorBuffer
from lib_sparse.src.functions import dense_to_tilesparse
from lib_sparse.src.sparse_matmul import AspB, AspRelu2B, spAB
from lib_sparse.src.triton_operators import unpack_batch_
```

The supported sparse value dtypes are `torch.bfloat16`, `torch.float8_e4m3fn`, and `torch.float8_e5m2`. In practice, `to_fp8` currently produces `float8_e4m3fn`.

## Compress an activation

`dense_to_tilesparse` accepts a two-dimensional tensor and has the current signature:

```python
dense_to_tilesparse(dense, scale, sparse_data=None, pack_sbit=False)
```

Pass `scale=None` for BF16 data or for an FP8 tensor whose scale is not needed by a subsequent sparse matmul:

```python
device = "cuda"
h = torch.randn(2048, 4096, device=device, dtype=torch.bfloat16).relu_()

sparse = dense_to_tilesparse(
    h,
    scale=None,
    pack_sbit=True,
)

print(sparse)
print("active values:", int(sparse.nnz().item()))
print("sparse bytes:", sparse.vram_size())
```

Compression is ReLU-oriented. The mask is computed from `dense > 0`; zero and negative entries are not stored. Therefore the input must already be non-negative if it is expected to round-trip. This behavior applies whether `pack_sbit` is false or true.

`pack_sbit=False` stores each active value in the input dtype. `pack_sbit=True` stores a bitstream: BF16 values use 15 bits each and FP8 values use 7 bits each because the sign bit is omitted. The `BitsparseTensor.vals` field is consequently a raw `uint8` buffer in packed mode.

## Decompress a complete tensor

There is no `decompress()` method on `BitsparseTensor`; use the Triton wrapper and supply the output tensor:

```python
def decompress(sparse):
    rows, columns = sparse.shape
    output = torch.empty(
        (rows, columns),
        device=sparse.vals.device,
        dtype=sparse.dtype,
    )
    return unpack_batch_(
        sparse,
        output,
        first_m_tile=0,
        grid_n=sparse.grid_n,
        K=columns,
        batch_rows=rows,
        num_tiles_in_batch=sparse.grid_m * sparse.grid_n,
    )

restored = decompress(sparse)
torch.testing.assert_close(restored, h, rtol=0, atol=0)
```

For an FP8-compressed activation, unpacking returns values in the stored FP8 domain. If `sparse.scale` is set, the original-domain values are obtained by converting the unpacked tensor to a suitable higher-precision dtype and multiplying by that scale. The sparse matmul helpers apply this scale internally, so manual rescaling is not needed when using those helpers.

## Matrix products involving sparse tensors

The sparse matmul helpers operate on two-dimensional matrices:

```python
# A: [M, N], sparse: [N, K] -> [M, K]
A = torch.randn(32, h.shape[0], device=device, dtype=torch.bfloat16)
y = AspB(A, sparse)

# sparse: [M, N], B: [N, K] -> [M, K]
B = torch.randn(h.shape[1], 128, device=device, dtype=torch.bfloat16)
y2 = spAB(sparse, B)
```

Both functions unpack the sparse operand to a dense temporary before multiplying. `AspB` can use an FP8 scale on the sparse right-hand operand. `spAB` can use an FP8 scale on the sparse left-hand operand.

`AspRelu2B(A, sparse)` is the ReLU² variant. It reconstructs `RELU2_SCALE * sparse_activation²` into BF16 before the matmul. The reconstruction is deliberately in BF16 because squaring FP8 values directly can overflow the FP8 range.

For lower peak VRAM, pass `row_batch > 0`. The implementation then unpacks tile-aligned row batches and accumulates each result. The batch size is rounded down to a whole number of 64-row tiles, with at least one tile per iteration:

```python
y = AspB(A, sparse, row_batch=512)
y2 = spAB(sparse, B, row_batch=512)
```

The blockwise path uses more launches and temporary work, so it is a memory/performance trade-off. For `spAB`, a preallocated `out` is required by the documented blockwise path if you want to control the destination; otherwise the helper creates one.

## Use the ReLU and ReLU² FFN layers

The high-level entry points are static `apply` methods rather than `nn.Module` objects:

```python
from lib_sparse.layers import FFNRelu, FFNRelu2

batch, d_in, d_ff, d_out = 8, 4096, 21504, 4096
x = torch.randn(batch, d_in, device=device, dtype=torch.bfloat16)
W1 = torch.randn(d_ff, d_in, device=device, dtype=torch.bfloat16)
W2 = torch.randn(d_out, d_ff, device=device, dtype=torch.bfloat16)

y_relu = FFNRelu.apply(
    x, W1, W2,
    sparse_data=None,
    pack_sbit=True,
    dtype=torch.bfloat16,
)

y_relu2 = FFNRelu2.apply(
    x, W1, W2,
    sparse_data=None,
    pack_sbit=True,
    storage_dtype=torch.bfloat16,
)
```

The actual matrix shapes required by the implementation are:

```text
x   : [*, d_in]
W1  : [d_ff, d_in]
W2  : [d_out, d_ff]
b1  : [d_ff]       optional
b2  : [d_out]      optional
out : [*, d_out]
```

`FFNRelu` computes `x -> x @ W1.T -> ReLU -> @ W2.T`. `FFNRelu2` computes `x -> x @ W1.T -> RELU2_SCALE * ReLU(...)² -> @ W2.T`. Leading batch dimensions are flattened for the two linear products and restored on return.

During forward, the activation is compressed and attached to the autograd context. During backward, the custom function computes the second-weight gradient through `AspB` or `AspRelu2B`, computes the input gradient with a dense/FP8 matmul, and applies a Triton ReLU mask or ReLU² derivative in place.

## Reuse one value buffer across layers

`TensorBuffer` is an append-only value allocation used by several compressions. It stores bytes, while its device-side `offset` records the next logical value position. This makes it useful for a stack of FFN layers where every layer's saved activation must remain alive until backward:

```python
from lib_sparse.bitsparse import TensorBuffer, bits_per_value

storage_dtype = torch.bfloat16
number_of_tensors = 8
worst_case_values = number_of_tensors * batch * d_ff

# Add up to seven logical padding values per tensor for packed alignment.
capacity_values = worst_case_values + 7 * number_of_tensors
buffer_bytes = (capacity_values * bits_per_value(storage_dtype) + 7) // 8

buffer = TensorBuffer(
    buffer_bytes,
    device=device,
    dtype=storage_dtype,
    pack_sbit=True,
)

buffer.reset_buffer()
saved_activations = []
for activation in activations:
    saved_activations.append(
        dense_to_tilesparse(
            activation,
            scale=None,
            sparse_data=buffer,
            pack_sbit=True,
        )
    )
```

For raw storage, allocate `worst_case_values * element_size` bytes and use `pack_sbit=False`. A shared buffer must use one consistent logical dtype and packing mode; do not mix BF16 and FP8 streams or packed and raw streams in the same buffer.

The buffer is not a general-purpose allocator. It does not grow or check capacity before a kernel writes. Allocate for the worst case, and inspect `buffer.offset` after a pass. Do not call `reset_buffer()` until every sparse tensor that refers to the buffer is no longer needed, including any autograd backward pass using it.

## FP8 workflow

`to_fp8` computes one scale for the whole tensor:

```python
from lib_sparse.fp8 import matmul, to_fp8

h_fp8, h_scale = to_fp8(h)
sparse_fp8 = dense_to_tilesparse(
    h_fp8,
    scale=h_scale,
    pack_sbit=True,
)
```

The scale maps the maximum absolute input value to the maximum finite value of the `float8_e4m3fn` format. `to_fp8` clamps very small scales to `1e-9` and currently emits `float8_e4m3fn`.

For `matmul`, BF16 inputs are converted to FP8 internally when `fp8=True`. Already-FP8 inputs must be accompanied by their tensor-wise scales. The output of the custom FP8 matmul is BF16:

```python
out = matmul(a, b, fp8=True, a_scale=a_scale, b_scale=b_scale)
```

`lib_sparse.fp8` selects a cuBLASLt-compatible layout at import time. Ada-class devices need a column-major layout for the right operand; Blackwell-class devices can use row-major `B` directly. If a machine contains different GPU capabilities, this device-0 decision is a limitation of the current implementation.

## Integrating with the Nemotron model

`nemotron/llm.py` adds four custom configuration fields:

```python
config.sparse_ffn   # route NemotronHMLP through FFNRelu2
config.use_ckpt     # dense fallback uses PyTorch checkpointing
config.sparse_data  # optional TensorBuffer
config.pack_sbit    # use the 7/15-bit stream
```

When `config.sparse_ffn` is true, `NemotronHMLP.forward` passes its `up_proj.weight` and `down_proj.weight` to `FFNRelu2.apply`. The sparse path is therefore appropriate for a `relu2` MLP with the usual `[intermediate, hidden]` up projection and `[hidden, intermediate]` down projection. The current Nemotron integration does not pass MLP biases into the sparse call; the model configuration normally uses `mlp_bias=False`.

## Current script/API mismatch

The experiment scripts are useful references, but some calls are from an older API. The current `dense_to_tilesparse` implementation requires `scale` and does not accept a `storage_dtype` keyword; for direct use, call it as `dense_to_tilesparse(tensor, scale=None, ...)` or preconvert with `to_fp8`. In particular, update the helper calls in `experiments/benchmark.py` before running them with the current library.
