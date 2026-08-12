import torch
from torch import Tensor

from lib_sparse.bitsparse import BitsparseTensor, TensorBuffer
from lib_sparse.src.functions import dense_to_tilesparse
from lib_sparse.src.triton_operators import unpack_batch_


SHAPES = ((128, 128), (129, 131))
SPARSITIES = (0.0, 0.5, 0.9, 1.0)


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
        dtype=sparse.value_dtype,
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
    torch.testing.assert_close(restored, dense, rtol=0, atol=0)
    assert int(sparse.nnz().item()) == int((dense > 0).sum().item())


def make_buffer(tensors: list[Tensor], pack_sbit: bool) -> TensorBuffer:
    """Allocate enough shared storage for every tensor's worst-case NNZ count."""
    capacity = sum(tensor.numel() for tensor in tensors)
    if pack_sbit:
        # Each allocation may align its logical value offset to eight values.
        capacity += 7 * len(tensors)
        size = (capacity * 15 + 7) // 8
    else:
        size = capacity * tensors[0].element_size()
    return TensorBuffer(
        size,
        device="cuda",
        dtype=torch.bfloat16,
        pack_sbit=pack_sbit,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("validate.py requires a CUDA device")

    generator = torch.Generator(device="cuda").manual_seed(0)
    passed = 0
    total_checks = len(SHAPES) * len(SPARSITIES) * 2 * 2

    for pack_sbit in (False, True):
        tensors = [
            generate_data(shape, generator, sparsity)
            for shape in SHAPES
            for sparsity in SPARSITIES
        ]

        # Validate standalone allocations.
        for dense in tensors:
            sparse = dense_to_tilesparse(dense, pack_sbit=pack_sbit)
            validate_sparse(dense, sparse)
            passed += 1

        # Validate several tensors sharing one preallocated value buffer. Keep
        # every sparse tensor alive and decompress after all writes so offsets
        # and non-overlapping allocations are covered.
        buffer = make_buffer(tensors, pack_sbit)
        compressed = [
            dense_to_tilesparse(dense, sparse_data=buffer, pack_sbit=pack_sbit)
            for dense in tensors
        ]
        for dense, sparse in zip(tensors, compressed):
            validate_sparse(dense, sparse)
            assert sparse.vals is buffer.vals
            passed += 1

        for shape in SHAPES:
            print(f"passed shape={shape}, pack_sbit={pack_sbit}")

    print(f"Passed: {passed}")
    print(f"Total checks: {total_checks}")


if __name__ == "__main__":
    main()
