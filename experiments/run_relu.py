import torch
import torch.nn.functional as F
import torch._logging
import math

from lib_sparse.layers import FFNRelu, FFNRelu_3
from lib_sparse.bitsparse import TensorBuffer

from .experiment import run_step, DeepFFN_abc
from .utils import setup_hooks

FFN_BLOCK_LAYERS = 2
LAYERS = 8
BATCH_SIZE = 10000
DIM = 4096

BASIC_MODE = True
PACK_15BIT = False

class DeepFFN(DeepFFN_abc):
    handles: list

    def __init__(self, layers, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, DIM, FFN_BLOCK_LAYERS)

    # @torch.compile
    def forward(self, x, buffer: TensorBuffer | None = None):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        if buffer is not None:
            buffer.reset_buffer()
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu.apply(x_inner, W1, W2, sparse_data=buffer, pack_15bit=PACK_15BIT)
                # x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu_3.apply(x_inner, W1, W2, W3, sparse_data=buffer, pack_15bit=PACK_15BIT)
        return x


def evaluate(bs=BATCH_SIZE, layers=LAYERS):
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = DeepFFN(layers, dtype=dtype)
    # if not BASIC_MODE:
    setup_hooks(model)

    # Run baseline
    if bs < 32_001:
        run_step(x, model, sparse=False, steps=2)
        tracking_dn, vram_dn, avg_time_dn = run_step(x, model, sparse=False, steps=3)
        print(f"Baseline: {vram_dn = :.0f} MB, avg_time = {avg_time_dn:.2f} ms")

    # Setup sparse buffer and run model (in basic mode layers allocate on-the-fly)
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.6 * (2 if FFN_BLOCK_LAYERS == 3 else 1)
        value_capacity = int(bs * hdim_expanded * layers * buffer_scale)
        bits_per_value = 15 if PACK_15BIT else 16
        buffer_size = (value_capacity * bits_per_value + 7) // 8
        buffer = TensorBuffer(
            buffer_size, dtype=dtype, device="cuda", pack_15bit=PACK_15BIT
        )

    run_step(x, model, buffer, sparse=True, steps=2)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=3)
    print(f"Compressed: {vram = :.0f} MB, avg_time = {avg_time:.2f} ms")

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-6, rtol=3e-6):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-6, rtol=3e-6)

    return vram_dn, avg_time_dn, vram, avg_time


def run_batch():
    print(f'{PACK_15BIT = }')
    import csv

    batch_sizes = [32, 128, 512, 2000, 4000, 8000, 16000, 32000, 40000, 75_000, 100_000]

    with open("relu_sparse.csv", "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "batch_size", "vram_dn", "avg_time_dn", "vram", "avg_time",
        ])

        for bs in batch_sizes:
            print("-" * 50)
            print(f'{bs = }')

            vram_dn, avg_time_dn, vram, avg_time = evaluate(bs=bs)
            writer.writerow([bs, vram_dn, avg_time_dn, vram, avg_time])
            f.flush()


if __name__ == "__main__":
    run_batch()
    # evaluate(bs=45_000)
