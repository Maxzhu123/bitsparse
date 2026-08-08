import torch
import torch._logging
import math

from lib_sparse.layers import FFNRelu, FFNRelu_3
from lib_sparse.bitsparse import TensorBuffer

from .experiment import run_step, DeepFFN_abc
from .utils import setup_hooks

FFN_BLOCK_LAYERS = 2
LAYERS = 16
BATCH_SIZE = 20000
DIM = 2048

BASIC_MODE = True
PACKED_15BIT = True

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
                x_inner = x
                x = x + FFNRelu.apply(x_inner, W1, W2, sparse_data=buffer, packed_15bit=PACKED_15BIT)
                # x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + FFNRelu_3.apply(x_inner, W1, W2, W3, sparse_data=buffer, packed_15bit=PACKED_15BIT)
        return x


def evaluate(bs=BATCH_SIZE, layers=LAYERS):
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = DeepFFN(layers, dtype=dtype)
    if not BASIC_MODE:
        setup_hooks(model)

    # Run baseline
    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time_dn = run_step(x, model, sparse=False, steps=1)
    print(f"Baseline: {vram_dn = :.0f} MB, avg_time = {avg_time_dn:.2f} ms")

    # Setup sparse buffer and run model (in basic mode layers allocate on-the-fly)
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.6 * (2 if FFN_BLOCK_LAYERS == 3 else 1)
        value_capacity = int(bs * hdim_expanded * layers * buffer_scale)
        bits_per_value = 15 if PACKED_15BIT else 16
        buffer_size = (value_capacity * bits_per_value + 7) // 8
        buffer = TensorBuffer(
            buffer_size, dtype=dtype, device="cuda", packed_15bit=PACKED_15BIT
        )

    run_step(x, model, buffer, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=2)
    print(f"Compressed: {vram = :.0f} MB, avg_time = {avg_time:.2f} ms")

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-6, rtol=3e-6):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-6, rtol=3e-6)

    return vram_dn, avg_time_dn, vram, avg_time


def run_batch():
    import csv

    batch_sizes = [32, 128, 512, 2000, 4000, 8000, 16000, 32000, 50000]

    with open("evaluate_sparse.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "batch_size", "vram_dn", "avg_time_dn", "vram", "avg_time",
        ])

        for bs in batch_sizes:
            vram_dn, avg_time_dn, vram, avg_time = evaluate(bs=bs)
            writer.writerow([bs, vram_dn, avg_time_dn, vram, avg_time])
            print("-" * 50)
            print(f'{bs = }')
            f.flush()


def run_base():
    """Configure deterministic/debug settings and launch the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)

    evaluate()


if __name__ == "__main__":
    # run_batch()
    evaluate(bs=5000)
