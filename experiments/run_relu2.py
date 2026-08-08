import torch
import torch.nn.functional as F
import math
import torch._logging

from lib_sparse.layers import FFNRelu2, FFNRelu2_3
from lib_sparse.bitsparse import TensorBuffer

from experiments.experiment import run_step, FFN_relu2_abc
from experiments.utils import setup_hooks

FFN_BLOCK_LAYERS = 2
LAYERS = 8
BATCH_SIZE = 10000
DIM = 4096

BASIC_MODE = True
PACKED_15BIT = True


class DeepFFN(FFN_relu2_abc):
    def __init__(self, dtype):
        super().__init__(dtype, LAYERS, DIM, FFN_BLOCK_LAYERS)

    def forward(self, x, buffer: TensorBuffer | None = None):
        """Run the sparse-activation FFN on ``x[B, D]`` through all residual layers."""
        if buffer is not None:
            buffer.reset_buffer()
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu2.apply(x_inner, W1, W2, sparse_data=buffer, packed_15bit=PACKED_15BIT)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu2_3.apply(x_inner, W1, W2, W3, sparse_data=buffer, packed_15bit=PACKED_15BIT)
        return x


def evaluate():
    dtype = torch.bfloat16
    # Setup model
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(BATCH_SIZE, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)
    model = DeepFFN(dtype=dtype)
    if not BASIC_MODE:
        setup_hooks(model)

    # Run baseline
    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=3)
    print(f'Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms')
    print("-" * 50)

    # Setup sparse buffer (in basic mode layers allocate on-the-fly)
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.55 * (2 if FFN_BLOCK_LAYERS == 3 else 1)
        value_capacity = int(BATCH_SIZE * hdim_expanded * LAYERS * buffer_scale)
        bits_per_value = 15 if PACKED_15BIT else 16
        buffer_size = (value_capacity * bits_per_value + 7) // 8
        buffer = TensorBuffer(
            buffer_size, dtype=dtype, device="cuda", packed_15bit=PACKED_15BIT
        )

    # Run sparse model
    run_step(x, model, buffer, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=3)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f'Total time: {avg_time:.2f} ms')

    print(f'{tracking_dn = }')
    print(f'{tracking = }')

    if not torch.allclose(tracking, tracking_dn, atol=3e-6, rtol=3e-6):
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-6, rtol=3e-6)
        assert vram < vram_dn * 1.1


def run_base():
    torch.set_printoptions(precision=7)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
