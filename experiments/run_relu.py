import torch
import torch.nn.functional as F
import torch._logging
import math

from lib_sparse.layers import FFNRelu, FFNRelu_3
from lib_sparse.bitsparse import TensorBuffer

from experiment import run_step, DeepFFN_abc, FFN
from utils import setup_hooks

FFN_BLOCK_LAYERS = 2
LAYERS = 8
BATCH_SIZE = 10000
DIM = 4096

CKPT = True
BASIC_MODE = True

class DeepFFN(DeepFFN_abc):
    handles: list

    def __init__(self, layers, sp_blocks, dtype, ckpt=CKPT):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, DIM, FFN_BLOCK_LAYERS, ckpt)

    # @torch.compile
    def forward(self, x, pack_15bit: bool, buffer: TensorBuffer|None = None):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        if buffer is not None:
            buffer.reset_buffer()
        if self.block_layers == 2:
            for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
                x_inner = F.rms_norm(x, x.shape[1:])
                if i < self.sp_blocks:
                    x = x + FFNRelu.apply(x_inner, W1, W2, sparse_data=buffer, pack_15bit=pack_15bit)
                    # x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
                else:
                    x = x + FFN.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu_3.apply(x_inner, W1, W2, W3, sparse_data=buffer, pack_15bit=pack_15bit)
        return x


def evaluate(bs=BATCH_SIZE, layers=LAYERS, sp_blocks=LAYERS):
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = DeepFFN(layers, sp_blocks, dtype=dtype)
    # if not BASIC_MODE:
    setup_hooks(model)

    # 1) Run baseline
    run_step(x, model, sparse=False, steps=2)
    tracking_dn, vram_dn, avg_time_dn = run_step(x, model, sparse=False, steps=5)
    print(f"Baseline: {vram_dn = :.0f} MB, avg_time = {avg_time_dn:.2f} ms")

    # 2) Setup sparse buffer and run model (in basic mode layers allocate on-the-fly)
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.55
        value_capacity = int(bs * hdim_expanded * layers * buffer_scale)
        bits_per_value = 16
        buffer_size = (value_capacity * bits_per_value + 7) // 8
        buffer = TensorBuffer(
            buffer_size, dtype=dtype, device="cuda", pack_15bit=False
        )

    run_step(x, model, buffer, sparse=True, steps=2)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, pack_15bit=False, steps=5)
    print(f"Compressed: {vram = :.0f} MB, avg_time = {avg_time:.2f} ms")

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-6, rtol=3e-6):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-6, rtol=3e-6)

    # 3) Run with 15bit storage
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.55
        value_capacity = int(bs * hdim_expanded * layers * buffer_scale)
        bits_per_value = 15
        buffer_size = (value_capacity * bits_per_value + 7) // 8
        buffer = TensorBuffer(
            buffer_size, dtype=dtype, device="cuda", pack_15bit=True
        )

    run_step(x, model, buffer, sparse=True, steps=2)
    tracking, vram_15bit, avg_time_15bit = run_step(x, model, buffer, sparse=True, pack_15bit=True, steps=5)
    print(f"Compressed 15bit: {vram_15bit = :.0f} MB, avg_time = {avg_time_15bit:.2f} ms")

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-6, rtol=3e-6):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-6, rtol=3e-6)


    return vram_dn, avg_time_dn, vram, avg_time, vram, avg_time_15bit


if __name__ == "__main__":
    from experiment import run_batch, run_layers
    # run_batch(evaluate, save_name="relu_ckpt.csv")
    run_layers(evaluate, bs=16_000, save_name="relu_layers.csv")
    # evaluate(bs=16_000)
