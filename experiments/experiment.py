from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import gc

from config import RELU2_SCALE
from experiments.utils import setup_hooks
from lib_sparse.bitsparse import TensorBuffer

LAYERS = 8
BATCH_SIZE = 10000
DIM = 4096

BASIC_MODE = True

# ------------------------------------------------------------------------------
# Evaluation Loop
# ------------------------------------------------------------------------------
def run_step(x, model, buffer=None, sparse=False, pack_15bit=False, steps=1):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")

    start = time.perf_counter()

    for _ in range(steps):
        x.grad = None
        model.zero_grad()
        torch.cuda.reset_peak_memory_stats("cuda")
        if sparse:
            y = model.forward(x, pack_15bit, buffer)
        else:
            y = model.forward_base(x)
        loss = (y - x).abs().mean()
        del y
        loss.backward()
        loss.detach()

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps

    # Get gradients
    with torch.no_grad():
        tracking = [loss.detach().cpu(), x.grad.std().cpu()]
        for n, p in model.named_parameters():
            if p.grad is not None:
                tracking.append(p.grad.std().cpu())
        tracking = torch.stack(tracking) * 1e3
        x.grad = None

    return tracking, allocated, avg_time


def run_batch(model_fn, save_name="results.csv"):
    import csv

    batch_sizes = [32, 128, 512, 2000, 4000, 8000, 16000, 32000]

    with open(save_name, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "batch_size", "vram_dn", "avg_time_dn", "vram", "avg_time", "vram_15bit", "avg_time_15bit",
        ])

        for bs in batch_sizes:
            print("-" * 50)
            print(f'{bs = }')

            vram_dn, avg_time_dn, vram, avg_time, vram_15bit, avg_time_15bit = evaluate(model_fn, bs=bs)
            writer.writerow([bs, vram_dn, avg_time_dn, vram, avg_time, vram_15bit, avg_time_15bit])
            f.flush()


def run_layers(model_fn, bs, save_name="results.csv"):
    import csv

    sp_blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    with open(save_name, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "sp_blocks", "vram_dn", "avg_time_dn", "vram", "avg_time", "vram_15bit", "avg_time_15bit",
        ])

        for b in sp_blocks:
            print("-" * 50)
            print(f'{b = }')

            vram_dn, avg_time_dn, vram, avg_time, vram_15bit, avg_time_15bit = evaluate(model_fn, bs=bs, sp_blocks=b)
            writer.writerow([b, vram_dn, avg_time_dn, vram, avg_time, vram_15bit, avg_time_15bit])
            f.flush()


def evaluate(model_fn, bs, layers=LAYERS, sp_blocks=LAYERS):
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = model_fn(layers, sp_blocks, dtype=dtype)
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

    run_step(x, model, buffer, sparse=True, pack_15bit=True, steps=2)
    tracking, vram_15bit, avg_time_15bit = run_step(x, model, buffer, sparse=True, pack_15bit=True, steps=5)
    print(f"Compressed 15bit: {vram_15bit = :.0f} MB, avg_time = {avg_time_15bit:.2f} ms")
    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-6, rtol=3e-6):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-6, rtol=3e-6)

    return vram_dn, avg_time_dn, vram, avg_time, vram_15bit, avg_time_15bit


# ------------------------------------------------------------------------------
# Generate parameters
# ------------------------------------------------------------------------------
def gen_params(dim, G, dtype, expansion=5.25, device="cuda"):
    """ 2 layer FFN parameters """
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)
    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    # Basic
    # shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    # W1 = W1 + 0.01 * shift * W1.std()
    # W2 = W2 - 0.01 * shift * W2.std()

    # Biased
    # W1 = W1 + 0.07 * W1.std()     # relu
    W1 = W1 + 0.1 * W1.std()        # relu2


    return W1, W2

# ------------------------------------------------------------------------------
# Baseline FFN layers parameters
# ------------------------------------------------------------------------------
class FFNRelu2_2(Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        z = x @ W1.T
        r = z.relu_()
        z = r.square()
        z.mul_(RELU2_SCALE)
        ctx.save_for_backward(x, W1, W2, r)
        return z @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, r = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        z = r.square().mul_(RELU2_SCALE)
        grad_W2 = grad_output.T @ z
        del z
        grad_z = grad_output @ W2
        grad_preact = grad_z * (2.0 * RELU2_SCALE * r)
        del grad_z, r
        grad_W1 = grad_preact.T @ x

        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_x = None
        if needs_x:
            grad_x = grad_preact @ W1
        return grad_x, grad_W1, grad_W2

    @staticmethod
    def apply_ckpt(x, W1, W2):
        return torch.utils.checkpoint.checkpoint(FFNRelu2_2.forward_ckpt, x, W1, W2, use_reentrant=False)

    @staticmethod
    def forward_ckpt(x, W1, W2):
        z = x @ W1.T
        r = z.relu_()
        z = r.square()
        z.mul_(RELU2_SCALE)
        return z @ W2.T


class FFN(Function):
    """Dense baseline autograd FFN for comparison.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = relu(x @ W1.T)`` and ``output = z @ W2.T``.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        """Run the dense FFN forward pass and save tensors for backward."""
        z = x @ W1.T
        z.relu_()
        output = z @ W2.T
        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute dense FFN gradients from ``grad_output[B, D]``."""
        x, W1, W2, z = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        grad_z = grad_output @ W2
        grad_W2 = grad_output.T @ z

        grad_preact = torch.ops.aten.threshold_backward.grad_input(
            grad_z, z, 0, grad_input=grad_z
        )
        del z, grad_z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_x = None
        if needs_x:
            grad_x = grad_preact @ W1

        grad_W1 = grad_preact.T @ x
        return grad_x, grad_W1, grad_W2, None, None

    @staticmethod
    def apply_ckpt(x, W1, W2):
        return torch.utils.checkpoint.checkpoint(FFN.forward_ckpt, x, W1, W2, use_reentrant=False)

    @staticmethod
    def forward_ckpt(x, W1, W2):
        """Run the dense FFN forward pass and save tensors for backward."""
        z = x @ W1.T
        z.relu_()
        output = z @ W2.T
        return output


# ------------------------------------------------------------------------------
# Sparse FFN base implementation
# ------------------------------------------------------------------------------
class FFNReluABC(nn.Module):
    """Stack of residual FFN layers ``x <- x + FFN(x)`` for benchmarking."""

    def __init__(self, dtype, layers, sp_blocks, hdim):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for _ in range(layers):
            W1, W2 = gen_params(hdim, G, dtype=dtype)
            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))
        self.sp_blocks = sp_blocks
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f'Model: {total_params = }, size={total_params * 2 // (1024 * 1024)} MB')

    # @torch.compile
    def forward_base(self, x):
        """Run the dense baseline on ``x[B, D]`` through all residual layers."""
        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            if i < self.sp_blocks:
                x = x + FFN.apply_ckpt(x_inner, W1, W2)
            else:
                x = x + FFN.apply(x_inner, W1, W2)
        return x


class FFNRelu2ABC(nn.Module):
    def __init__(self, dtype, sp_blocks, layers=12, hidm=4096):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for _ in range(layers):
            W1, W2 = gen_params(hidm, G, dtype=dtype)
            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))
        self.sp_blocks = sp_blocks

    def forward_base(self, x):
        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            if i < self.sp_blocks:
                x = x + FFNRelu2_2.apply_ckpt(x_inner, W1, W2)
            else:
                x = x + FFNRelu2_2.apply(x_inner, W1, W2)
        return x


