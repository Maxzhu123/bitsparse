import torch
import torch.nn.functional as F
import math
import torch._logging

from lib_sparse.layers import FFNRelu2, FFNRelu2_3, FFNSparseRelu2
from lib_sparse.bitsparse import TensorBuffer

from experiments.experiment import run_step, FFN_relu2_abc
from experiments.utils import setup_hooks

LAYERS = 8
BATCH_SIZE = 10000
DIM = 4096

BASIC_MODE = True
PACK_15BIT = True


class DeepFFN(FFN_relu2_abc):
    def __init__(self, layers, dtype):
        super().__init__(dtype, layers, DIM, 2)

    def forward(self, x, buffer: TensorBuffer | None = None):
        """Run the sparse-activation FFN on ``x[B, D]`` through all residual layers."""
        if buffer is not None:
            buffer.reset_buffer()
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu2.apply(x_inner, W1, W2, sparse_data=buffer, pack_15bit=PACK_15BIT)
                # x = x + FFNSparseRelu2.apply(x_inner, W1, W2, buffer, PACK_15BIT)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu2_3.apply(x_inner, W1, W2, W3, sparse_data=buffer, pack_15bit=PACK_15BIT)
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
        tracking_dn, vram_dn, avg_time_dn = run_step(x, model, sparse=False, steps=5)
        print(f"Baseline: {vram_dn = :.0f} MB, avg_time = {avg_time_dn:.2f} ms")

    # Setup sparse buffer and run model (in basic mode layers allocate on-the-fly)
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.55
        value_capacity = int(bs * hdim_expanded * layers * buffer_scale)
        bits_per_value = 15 if PACK_15BIT else 16
        buffer_size = (value_capacity * bits_per_value + 7) // 8
        buffer = TensorBuffer(
            buffer_size, dtype=dtype, device="cuda", pack_15bit=PACK_15BIT
        )

    run_step(x, model, buffer, sparse=True, steps=2)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=5)
    print(f"Compressed: {vram = :.0f} MB, avg_time = {avg_time:.2f} ms")

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-6, rtol=3e-6):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-6, rtol=3e-6)

    return vram_dn, avg_time_dn, vram, avg_time


if __name__ == "__main__":
    from experiment import run_batch
    run_batch(evaluate, save_name="relu2_15bit.csv")
    # evaluate(bs=20_000)
