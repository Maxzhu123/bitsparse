from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import gc

from lib_sparse.bitsparse import RELU2_SCALE


# ------------------------------------------------------------------------------
# Evaluation Loop
# ------------------------------------------------------------------------------
def run_step(x, model, buffer=None, sparse=False, steps=1):
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
            y = model.forward(x, buffer)
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

def run_batch(evaluate, save_name="results.csv"):
    import csv

    batch_sizes = [32, 128, 512, 2000, 4000, 8000, 16000, 32000]

    with open(save_name, "a", newline="") as f:
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

    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01 * shift * W1.std()
    W2 = W2 - 0.01 * shift * W2.std()
    return W1, W2


def gen_params_3(dim, G, dtype, expansion=5.25, device="cuda"):
    """ 3 layer FFN parameters """

    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    W2 = torch.empty(hdim, hdim, device=device, dtype=dtype)
    W3 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)
    torch.nn.init.xavier_uniform_(W2, generator=G)
    torch.nn.init.xavier_uniform_(W3, generator=G)
    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01 * shift * W1.std()
    W2 = W2 + 0.01 * shift * W2.std()
    W3 = W3 - 0.01 * shift * W3.std()
    return W1, W2, W3


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


class FFNRelu2_3(Function):
    """Dense 3-layer FFN with ReLU-squared activations — in-place forward, saved relu masks."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T              # pre-activation 1  [B, H]
        r1 = z1.relu_()            # r1 = relu(z1), in-place
        z1 = r1.square()
        z1.mul_(RELU2_SCALE)       # z1 = k * r1^2

        z2 = z1 @ W2.T             # pre-activation 2  [B, H]
        r2 = z2.relu_()            # r2 = relu(z2), in-place
        z2 = r2.square()
        z2.mul_(RELU2_SCALE)       # z2 = k * r2^2

        ctx.save_for_backward(x, r1, r2, W1, W2, W3)
        return z2 @ W3.T            # output [B, D]

    @staticmethod
    def backward(ctx, grad_output):
        x, r1, r2, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        # layer 3: linear
        grad_a2 = grad_output @ W3                                # [B, H]
        grad_W3 = grad_output.T @ (r2.square().mul_(RELU2_SCALE)) # [D, H]
        # relu2 backward: d/d(r2) of k*r2^2 = 2*k*r2
        grad_z2 = grad_a2 * (2.0 * RELU2_SCALE * r2)
        del r2
        # layer 2: linear
        grad_a1 = grad_z2 @ W2                                    # [B, H]
        grad_W2 = grad_z2.T @ (r1.square().mul_(RELU2_SCALE))    # [H, H]
        del grad_z2, grad_a2

        # relu1 backward: d/d(r1) of k*r1^2 = 2*k*r1
        grad_z1 = grad_a1 * (2.0 * RELU2_SCALE * r1)
        del grad_a1, r1
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()
        # layer 1: linear
        grad_x = grad_z1 @ W1 if needs_x else None
        grad_W1 = grad_z1.T @ x                                   # [H, D]
        del grad_z1

        return grad_x, grad_W1, grad_W2, grad_W3


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


class FFN_3(Function):
    """Dense 3-layer FFN — normal backward (saved intermediates, no checkpointing)."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T          # pre-activation 1  [B, H]
        z1.relu_()              # in-place ReLU → z1 is now post-activation
        z2 = z1 @ W2.T          # pre-activation 2  [B, H]
        z2.relu_()              # in-place ReLU → z2 is now post-activation
        output = z2 @ W3.T      # output           [B, D]
        ctx.save_for_backward(x, z1, z2, W1, W2, W3)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, a1, a2, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]
        # a1, a2 are already post-activation (relu_ was in-place)

        # layer 3: linear
        grad_a2 = grad_output @ W3          # [B, H]
        grad_W3 = grad_output.T @ a2        # [D, H]
        # relu2 backward — fused CUDA kernel, faster than torch.where
        grad_z2 = torch.ops.aten.threshold_backward.grad_input(
            grad_a2, a2, 0, grad_input=grad_a2
        )
        del grad_a2, a2
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()
        # layer 2: linear
        grad_a1 = grad_z2 @ W2              # [B, H]
        grad_W2 = grad_z2.T @ a1            # [H, H]

        del grad_z2
        # relu1 backward
        grad_z1 = torch.ops.aten.threshold_backward.grad_input(
            grad_a1, a1, 0, grad_input=grad_a1
        )
        del grad_a1, a1
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        # layer 1: linear
        grad_x = grad_z1 @ W1 if needs_x else None
        grad_W1 = grad_z1.T @ x             # [H, D]

        return grad_x, grad_W1, grad_W2, grad_W3


# ------------------------------------------------------------------------------
# Sparse FFN base implementation
# ------------------------------------------------------------------------------
class DeepFFN_abc(nn.Module):
    """Stack of residual FFN layers ``x <- x + FFN(x)`` for benchmarking."""

    def __init__(self, dtype, layers, hdim, block_layers=2):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.block_layers = block_layers
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for _ in range(layers):
            if self.block_layers == 2:
                W1, W2 = gen_params(hdim, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
            else:
                W1, W2, W3 = gen_params_3(hdim, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
                self.W3s.append(nn.Parameter(W3))
        if self.block_layers == 3:
            self.block_forward = FFN_3.apply
        elif self.block_layers == 2:
            self.block_forward = FFN.apply
        else:
            raise NotImplementedError

        # total_params = sum(p.numel() for p in self.parameters())
        # print(f'Model: {total_params = }, size={total_params * 2 // (1024 * 1024)} MB')

    # @torch.compile
    def forward_base(self, x):
        """Run the dense baseline on ``x[B, D]`` through all residual layers."""
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + self.block_forward(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + self.block_forward(x_inner, W1, W2, W3)
        return x


class FFN_relu2_abc(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096, block_layers=2):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.block_layers = block_layers
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for _ in range(layers):
            if self.block_layers == 2:
                W1, W2 = gen_params(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
            else:
                W1, W2, W3 = gen_params_3(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
                self.W3s.append(nn.Parameter(W3))
        if self.block_layers == 3:
            self.block_forward = FFNRelu2_3.apply
        elif self.block_layers == 2:
            self.block_forward = FFNRelu2_2.apply
        else:
            raise NotImplementedError

    def forward_base(self, x):
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + self.block_forward(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + self.block_forward(x_inner, W1, W2, W3)
        return x


