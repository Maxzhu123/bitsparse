from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import gc
from cprint import c_print
import os

from experiments.utils import setup_hooks
from lib_sparse.bitsparse import TensorBuffer, bits_per_value
from lib_sparse.config import RELU2_SCALE
from lib_sparse.fp8 import is_fp8, matmul, to_fp8

LAYERS = 6
BATCH_SIZE = 10000
DIM = 4096

# Datatype for matmul + activation caching: torch.bfloat16 or torch.float8_e4m3fn.
DTYPE = torch.float8_e4m3fn # torch.bfloat16 #

# Correctness tolerance: exact for bf16, loose for the fp8 quantization error.
CHECK_RTOL = CHECK_ATOL = 3e-6 if DTYPE == torch.bfloat16 else 1e-1

BASIC_MODE = False
DATA_SPARSITY = "Sparse"        # "Normal", "Sparse", "ReLU"
c_print(f'{DATA_SPARSITY = }', color="green")
# ------------------------------------------------------------------------------
# Evaluation Loop
# ------------------------------------------------------------------------------
def run_step(x, model, buffer=None, sparse=False, pack_sbit=False,
             storage_dtype=torch.bfloat16, steps=1):
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
            y = model.forward(x, pack_sbit, buffer, storage_dtype)
        else:
            y = model.forward_base(x)
        loss = (y - x).abs().mean()
        del y
        loss.backward()
        loss.detach()

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / max(steps, 1)

    # Get gradients
    with torch.no_grad():
        tracking = [loss.detach().cpu(), x.grad.std().cpu()]
        for n, p in model.named_parameters():
            if p.grad is not None:
                tracking.append(p.grad.std().cpu())
        tracking = torch.stack(tracking) * 1e3
        x.grad = None

    return tracking, allocated, avg_time


def run_batch(model_fn, warmup_steps, eval_steps, batch_sizes=None, sp_blocks=LAYERS, save_name="results.csv"):
    import csv

    if batch_sizes is None:
        batch_sizes = [32, 128, 512, 2000, 4000, 8000, 16000, 32000]

    file_exists = os.path.exists(save_name)

    with open(save_name, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "batch_size", "vram_base", "vram", "vram_15bit", "vram_ckpt", "avg_time_base", "avg_time_dn", "avg_time", "avg_time_15bit",
            ])

        for bs in batch_sizes:
            print("-" * 50)
            print(f'{bs = }')

            vram_base, avg_time_base, vram_ckpt, avg_time_ckpt, vram, avg_time, vram_15bit, avg_time_15bit = evaluate(
                model_fn, sp_blocks=sp_blocks, eval_steps=eval_steps, warmup_steps=warmup_steps, bs=bs)
            writer.writerow([bs, vram_base, vram, vram_15bit, vram_ckpt, avg_time_base, avg_time, avg_time_15bit, avg_time_ckpt])
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

            vram_base, avg_time_base, vram_ckpt, avg_time_ckpt, vram, avg_time, vram_15bit, avg_time_15bit = evaluate(model_fn, bs=bs, sp_blocks=b)
            writer.writerow([bs, vram_base, vram, vram_15bit, vram_ckpt, avg_time_base, avg_time, avg_time_15bit, avg_time_ckpt])
            f.flush()


def evaluate(model_fn, bs, warmup_steps=1, eval_steps=5, layers=LAYERS, sp_blocks=LAYERS):
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    dtype = torch.bfloat16
    storage_dtype = DTYPE
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = model_fn(layers, 0, dim=DIM, dtype=dtype)
    # if not BASIC_MODE:
    setup_hooks(model)

    # 0) Run baseline model
    run_step(x, model, sparse=False, steps=warmup_steps)
    tracking_dn, vram_base, avg_time_base = run_step(x, model, sparse=False, steps=eval_steps)
    print(f"Baseline: {vram_base = :.0f} MB, avg_time = {avg_time_base:.2f} ms")

    # 1) Run checkpointed
    model.sp_blocks = sp_blocks
    run_step(x, model, sparse=False, steps=warmup_steps)
    tracking_dn, vram_dn, avg_time_dn = run_step(x, model, sparse=False, steps=eval_steps)
    print(f"Checkpointed: {vram_dn = :.0f} MB, avg_time = {avg_time_dn:.2f} ms")

    # 2) Setup sparse buffer and run model (in basic mode layers allocate on-the-fly)
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.55
        value_capacity = int(bs * hdim_expanded * layers * buffer_scale)
        buffer_size = value_capacity * torch.empty((), dtype=storage_dtype).element_size()
        buffer = TensorBuffer(
            buffer_size, dtype=storage_dtype, device="cuda", pack_sbit=False
        )

    run_step(x, model, buffer, sparse=True, storage_dtype=storage_dtype, steps=warmup_steps)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, pack_sbit=False,
                                        storage_dtype=storage_dtype, steps=eval_steps)
    print(f"Compressed: {vram = :.0f} MB, avg_time = {avg_time:.2f} ms")
    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=CHECK_ATOL, rtol=CHECK_RTOL):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=CHECK_ATOL, rtol=CHECK_RTOL)

    # 3) Run with bit-packed storage
    buffer = None
    if not BASIC_MODE:
        hdim_expanded = math.floor(DIM * 5.25)
        buffer_scale = 0.55
        value_capacity = int(bs * hdim_expanded * layers * buffer_scale)
        buffer_size = (value_capacity * bits_per_value(storage_dtype) + 7) // 8
        buffer = TensorBuffer(
            buffer_size, dtype=storage_dtype, device="cuda", pack_sbit=True
        )

    run_step(x, model, buffer, sparse=True, pack_sbit=True,
             storage_dtype=storage_dtype, steps=warmup_steps)
    tracking, vram_15bit, avg_time_15bit = run_step(x, model, buffer, sparse=True, pack_sbit=True,
                                                    storage_dtype=storage_dtype, steps=eval_steps)
    print(f"Compressed 15bit: {vram_15bit = :.0f} MB, avg_time = {avg_time_15bit:.2f} ms")
    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=CHECK_ATOL, rtol=CHECK_RTOL):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=CHECK_ATOL, rtol=CHECK_RTOL)

    return vram_base, avg_time_base, vram_dn, avg_time_dn, vram, avg_time, vram_15bit, avg_time_15bit


def evaluate_nobase(model_fn, bs, warmup_steps=1, eval_steps=5, layers=LAYERS, sp_blocks=LAYERS):
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = model_fn(layers, sp_blocks, dim=DIM, dtype=dtype)
    # if not BASIC_MODE:
    setup_hooks(model)

    # 2) Setup sparse buffer and run model (in basic mode layers allocate on-the-fly)
    run_step(x, model, None, sparse=True, storage_dtype=DTYPE, steps=warmup_steps)
    tracking, vram, avg_time = run_step(x, model, None, sparse=True, pack_sbit=False,
                                        storage_dtype=DTYPE, steps=eval_steps)
    print(f"Compressed: {vram = :.0f} MB, avg_time = {avg_time:.2f} ms")

    return vram, avg_time



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
    if DATA_SPARSITY == "Normal":
        shift = torch.randn(1, generator=G, device=device, dtype=dtype)
        W1 = W1 + 0.01 * shift * W1.std()
        W2 = W2 - 0.01 * shift * W2.std()
    elif DATA_SPARSITY == "Sparse":
        W1 = W1 + 0.1 * W1.std()        # 80% sparsity with ReLU2
    elif DATA_SPARSITY == "ReLU":
        W1 = W1 + 0.1 * W1.std()        # 80% sparsity with ReLU
    else:
        raise NotImplementedError("Unknown sparsity type")


    return W1, W2

# ------------------------------------------------------------------------------
# Baseline FFN layers parameters
# ------------------------------------------------------------------------------
class FFNRelu2_2(Function):
    """Dense baseline autograd FFN with ReLU² activation for comparison.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = RELU2_SCALE * relu(x @ W1.T)²`` and ``output = z @ W2.T``.  The
    forward matmuls are quantized to FP8 (fp8 + fp8 -> bf16) while activations
    stay in BF16, matching the ``lib_sparse.layers`` FFN.
    """
    @staticmethod
    def forward(ctx, x, W1, W2):
        fp8 = is_fp8(DTYPE)
        z = matmul(x, W1.T, fp8)
        r = z.relu_()
        z = r.square()
        z.mul_(RELU2_SCALE)
        ctx.save_for_backward(x, W1, W2, r)
        return matmul(z, W2.T, fp8)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, r = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]
        fp8 = is_fp8(DTYPE)

        # Use fp8 grad_output
        if fp8:
            grad_output, scale = to_fp8(grad_output)
        else:
            grad_output, scale = grad_output, None

        z = r.square().mul_(RELU2_SCALE)
        grad_W2 = matmul(grad_output.T, z, fp8, a_scale=scale)
        del z
        grad_z = matmul(grad_output, W2, fp8, a_scale=scale)
        grad_preact = grad_z * (2.0 * RELU2_SCALE * r)
        del grad_z, r

        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        # Use fp8 grad_preact
        if fp8:
            grad_preact, scale = to_fp8(grad_preact)
        else:
            grad_preact, scale = grad_preact, None

        grad_x = None
        if needs_x:
            grad_x = matmul(grad_preact, W1, fp8, a_scale=scale)
        grad_W1 = matmul(grad_preact.T, x, fp8, a_scale=scale)
        return grad_x, grad_W1, grad_W2

    @staticmethod
    def apply_ckpt(x, W1, W2):
        return torch.utils.checkpoint.checkpoint(FFNRelu2_2.forward_ckpt, x, W1, W2, use_reentrant=False)

    @staticmethod
    def forward_ckpt(x, W1, W2):
        """Run the dense FFN forward pass and save tensors for backward."""
        fp8 = is_fp8(DTYPE)
        z = matmul(x, W1.T, fp8)
        r = z.relu_()
        z = r.square()
        z.mul_(RELU2_SCALE)
        return matmul(z, W2.T, fp8)


class FFN(Function):
    """Dense baseline autograd FFN for comparison.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = relu(x @ W1.T)`` and ``output = z @ W2.T``.  The forward matmuls
    are quantized to FP8 (fp8 + fp8 -> bf16) while activations stay in BF16,
    matching the ``lib_sparse.layers`` FFN.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        """Run the dense FFN forward pass and save tensors for backward."""
        fp8 = is_fp8(DTYPE)
        z = matmul(x, W1.T, fp8)
        z.relu_()
        output = matmul(z, W2.T, fp8)
        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute dense FFN gradients from ``grad_output[B, D]``."""
        x, W1, W2, z = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]
        fp8 = is_fp8(DTYPE)

        # Use fp8 grad_output
        if fp8:
            grad_output, scale = to_fp8(grad_output)
        else:
            grad_output, scale = grad_output, None

        grad_z = matmul(grad_output, W2, fp8, a_scale=scale)
        grad_W2 = matmul(grad_output.T, z, fp8, a_scale=scale)

        grad_preact = torch.ops.aten.threshold_backward.grad_input(
            grad_z, z, 0, grad_input=grad_z
        )
        del z, grad_z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        # Use fp8 grad_preact
        if fp8:
            grad_preact, scale = to_fp8(grad_preact)
        else:
            grad_preact, scale = grad_preact, None
            
        if needs_x:
            grad_x = matmul(grad_preact, W1, fp8, a_scale=scale)
        else:
            grad_x = None
        grad_W1 = matmul(grad_preact.T, x, fp8, a_scale=scale)
        return grad_x, grad_W1, grad_W2, None, None

    @staticmethod
    def apply_ckpt(x, W1, W2):
        return torch.utils.checkpoint.checkpoint(FFN.forward_ckpt, x, W1, W2, use_reentrant=False)

    @staticmethod
    def forward_ckpt(x, W1, W2):
        """Run the dense FFN forward pass and save tensors for backward."""
        fp8 = is_fp8(DTYPE)
        z = matmul(x, W1.T, fp8)
        z.relu_()
        output = matmul(z, W2.T, fp8)
        return output


# ------------------------------------------------------------------------------
# Sparse FFN base implementation
# ------------------------------------------------------------------------------
class FFNReluABC(nn.Module):
    """Stack of residual FFN layers ``x <- x + FFN(x)`` for benchmarking."""

    def __init__(self, dtype, layers, sp_blocks, hdim):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
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
    def __init__(self, dtype, layers, sp_blocks, hdim):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
        for _ in range(layers):
            W1, W2 = gen_params(hdim, G, dtype=dtype)
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
