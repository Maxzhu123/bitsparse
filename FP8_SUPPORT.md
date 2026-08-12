# FP8 Support Implementation Brief

Add FP8 support to `lib_sparse` for compressed saved activations. Keep normal
forward computation and reconstructed tensors in BF16; FP8 is initially a
storage format, not an end-to-end FP8 matmul path.

## Instructions

1. Add separate storage and output dtype metadata to `BitsparseTensor`.
   Preserve BF16 as the default and avoid breaking existing callers.
2. Extend `dense_to_tilesparse` and `TensorBuffer` so BF16 input values can be
   compacted into FP8 storage. Validate buffer dtype, device, packing mode, and
   capacity before writing.
3. Support `torch.float8_e4m3fn` first. Add a scale to the sparse metadata and
   quantize values before storage. Compute the bitmask from the original BF16
   values so positive values rounded to zero do not lose their ReLU mask.
4. Update the Triton compact/unpack kernels to write FP8 and reconstruct BF16.
   Promote values before scaling, squaring, or applying the ReLU-squared
   derivative.
5. Update `AspB`, `spAB`, and `AspRelu2B` to allocate reconstructed matrices in
   the output/compute dtype rather than the storage dtype.
6. Reject FP8 combined with `pack_sbit=True` for the first implementation. The
   current 15-bit codec is specific to BF16 and must not be reused for FP8.
7. Feature-test native FP8 support in PyTorch, Triton, and the current GPU.
   On unsupported devices, provide a clear capability error while retaining
   all existing BF16 behavior.
8. Extend `experiments/validate.py` with FP8 round-trip checks for one regular
   and one irregular shape, standalone and preallocated buffers, multiple
   sparsities, correct NNZ counts, and non-overlapping shared-buffer offsets.
   Use an accuracy tolerance appropriate for scaled FP8 instead of exact
   equality.
9. Run the existing BF16 raw and 15-bit validation cases to confirm there are
   no regressions. Add focused checks for invalid dtype/packing combinations.

Keep the API changes small, document the scale convention, and benchmark only
after correctness passes.
