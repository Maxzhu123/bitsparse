import os

import torch
from torch import Tensor

from lib_sparse.bitsparse import BitsparseTensor, TensorBuffer, bits_per_value
from lib_sparse.src.functions import dense_to_tilesparse
from lib_sparse.src.triton_operators import unpack_batch_


SHAPES = ((128, 128), (129, 131))
SPARSITIES = (0.0, 0.5, 0.9, 1.0)
VALIDATION_DTYPE = getattr(
    torch, os.environ.get("VALIDATION_DTYPE", "bfloat16")
)


def generate_data(
    shape: tuple[int, int],
    generator: torch.Generator,
    sparsity: float,
) -> Tensor:
    """Create non-negative BF16 data with a known proportion of zeros."""
    dense = torch.randn(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).abs_()
    zero_mask = torch.rand(shape, generator=generator, device="cuda") < sparsity
    dense.masked_fill_(zero_mask, 0)
    return dense


def decompress(sparse: BitsparseTensor) -> Tensor:
    """Decompress a complete BitsparseTensor into a new dense tensor."""
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


def validate_sparse(dense: Tensor, sparse: BitsparseTensor) -> None:
    restored = decompress(sparse)
    if sparse.scale is not None:
        # Scaled FP8 rounds to the nearest representable step, so compare within
        # the quantization error (eps is 0.125 for e4m3fn, 0.25 for e5m2).
        eps = torch.finfo(sparse.dtype).eps
        torch.testing.assert_close(
            restored,
            dense,
            rtol=eps,
            atol=sparse.scale.item() * eps,
        )
    else:
        torch.testing.assert_close(restored, dense, rtol=0, atol=0)
    assert int(sparse.nnz().item()) == int((dense.float() > 0).sum().item())


def make_buffer(
    tensors: list[Tensor],
    storage_dtype: torch.dtype,
    pack_sbit: bool,
) -> TensorBuffer:
    """Allocate enough shared storage for every tensor's worst-case NNZ count."""
    capacity = sum(tensor.numel() for tensor in tensors)
    if pack_sbit:
        # Each allocation may align its logical value offset to eight values.
        capacity += 7 * len(tensors)
        size = (capacity * bits_per_value(storage_dtype) + 7) // 8
    else:
        size = capacity * torch.empty((), dtype=storage_dtype).element_size()
    return TensorBuffer(
        size,
        device="cuda",
        dtype=storage_dtype,
        pack_sbit=pack_sbit,
    )


def compress(
    dense: Tensor,
    storage_dtype: torch.dtype,
    pack_sbit: bool,
    buffer: TensorBuffer | None = None,
) -> BitsparseTensor:
    kwargs = {"sparse_data": buffer, "pack_sbit": pack_sbit}
    if storage_dtype != torch.bfloat16:
        kwargs["storage_dtype"] = storage_dtype
    return dense_to_tilesparse(dense, scale=None, **kwargs)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("validate.py requires a CUDA device")

    generator = torch.Generator(device="cuda").manual_seed(0)
    passed = 0
    pack_sbit_options = (False, True)
    total_checks = (
        len(SHAPES) * len(SPARSITIES) * len(pack_sbit_options) * 2
    )

    for pack_sbit in pack_sbit_options:
        tensors = [
            generate_data(shape, generator, sparsity)
            for shape in SHAPES
            for sparsity in SPARSITIES
        ]

        # Validate standalone allocations.
        for dense in tensors:
            sparse = compress(dense, VALIDATION_DTYPE, pack_sbit)
            validate_sparse(dense, sparse)
            passed += 1

        # Validate several tensors sharing one preallocated value buffer. Keep
        # every sparse tensor alive and decompress after all writes so offsets
        # and non-overlapping allocations are covered.
        buffer = make_buffer(tensors, VALIDATION_DTYPE, pack_sbit)
        compressed = [
            compress(dense, VALIDATION_DTYPE, pack_sbit, buffer)
            for dense in tensors
        ]
        for dense, sparse in zip(tensors, compressed):
            validate_sparse(dense, sparse)
            assert sparse.vals is buffer.vals
            passed += 1

        for shape in SHAPES:
            print(
                f"passed shape={shape}, storage_dtype={VALIDATION_DTYPE}, "
                f"pack_sbit={pack_sbit}"
            )

    print(f"Passed: {passed}")
    print(f"Total checks: {total_checks}")


if __name__ == "__main__":
    main()
