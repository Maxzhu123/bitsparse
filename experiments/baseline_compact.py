import math
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import gc
import time

from lib_sparse.fp8 import matmul, to_fp8


# Storage precision for the compressed activation projection.
#   True  -> quantize the projected activation to fp8 + scale before saving
#            (half the saved gradient-projection memory; fp8 matmuls).
#   False -> save the raw bf16 projection; bf16 matmuls.
USE_FP8 = False


def setup_hooks(model: nn.Module):
    """ Simulate hook optimiser that applies update + clears grads immediately."""
    def hook(w):
        if hasattr(w, "small_grad"):
            # Decode gradient
            grad_W = w.projector.decode(
                w.small_grad
            )
            w.small_grad = None

        w.grad = None
        return

    model.handles = []
    for n, p in model.named_parameters():
        handle = p.register_post_accumulate_grad_hook(hook)
        model.handles.append(handle)


class ReLUSquaredW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(x, W)

        # Avoid an extra temporary from relu(x) ** 2
        z = x.clamp_min(0)
        z = z.square()
        return z @ W.T

    @staticmethod
    def backward(ctx, grad_output):
        x, W = ctx.saved_tensors

        # Gradient through y = z @ W.T
        grad_z = grad_output @ W

        # d/dx ReLU(x)^2 = 2 * ReLU(x)
        grad_x = grad_z * (2 * x.clamp_min(0))

        # Recompute z for grad_W
        z = x.clamp_min(0)
        z = z.square()

        # grad_W: [out_features, in_features]
        grad_W = grad_output.T @ z

        return grad_x, grad_W


# ---------------------------------------------------------
# Projection + gradient decoder
# ---------------------------------------------------------
class GaussianProjector(nn.Module):
    """
    R: [in_features, rank]

    encode:  x       -> x @ R
    decode:  g_small -> g_small @ R.T

    In fp8 mode the random projection matrix R is re-generated from the same
    seed (so project and decode share the exact same R), and the projection
    matmuls go through the fp8 path (fp8 + fp8 -> bf16).
    """
    def __init__(self, in_features, rank):
        super().__init__()

        self.in_features = in_features
        self.rank = rank

    @torch.compiler.disable()
    def _make_R(self):
        g = torch.Generator(device="cuda")
        g.manual_seed(self.seed)

        R = torch.randn(self.in_features, self.rank, generator=g, device="cuda", dtype=torch.bfloat16)
        R /= math.sqrt(self.rank)

        # In fp8 mode pre-quantize R to fp8 + scale (consistent for encode/decode).
        if USE_FP8:
            R_fp8, R_scale = to_fp8(R)
            return R_fp8, R_scale
        return R, None

    @torch.compiler.disable()
    def project(self, x, x_fp8=None, x_scale=None):
        self.seed = random.randint(0, 2**16)
        R, R_scale = self._make_R()

        if USE_FP8:
            # Reuse the pre-quantized fp8 x when provided (avoids re-quantizing
            # x for the projection matmul); otherwise quantize on the fly.
            if x_fp8 is None:
                x_fp8, x_scale = to_fp8(x)
            return matmul(x_fp8, R, True, a_scale=x_scale, b_scale=R_scale)
        return x @ R

    @torch.compiler.disable()
    def decode(self, grad_small):
        # Same seed as project -> identical R.
        R, R_scale = self._make_R()

        if USE_FP8:
            # grad_small is fp8 already (saved by _CompActLinear); R fp8.
            return matmul(grad_small, R.T, True, b_scale=R_scale)
        return grad_small @ R.T


# ---------------------------------------------------------
# Custom autograd operation
# ---------------------------------------------------------
class _CompActLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, projector):
        weight.projector = projector

        if USE_FP8:
            x_fp8, x_scale = to_fp8(x)
        else:
            x_fp8, x_scale = x, None

        y = matmul(x_fp8, weight.T, USE_FP8, a_scale=x_scale)

        # save compressed x rather than full x.
        x_small = projector.project(x, x_fp8=x_fp8, x_scale=x_scale)
        # Quantize the saved projection to fp8 + scale (halves its memory).
        if USE_FP8:
            x_small_fp8, x_small_scale = to_fp8(x_small)
        else:
            x_small_fp8, x_small_scale = x_small, None
        ctx.x_small_scale = x_small_scale
        ctx.save_for_backward(x_small_fp8, weight)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_small_fp8, weight = ctx.saved_tensors
        x_small_scale = ctx.x_small_scale

        # Dequantize the saved projection back to bf16 for the gradient matmul.
        if x_small_scale is not None:
            x_small = x_small_fp8.to(torch.bfloat16) * x_small_scale
        else:
            x_small = x_small_fp8

        # Gradient w.r.t. activation is still exact.
        grad_x = matmul(grad_output, weight, USE_FP8)

        grad_weight_small = matmul(grad_output.T, x_small, USE_FP8)

        weight.small_grad = grad_weight_small
        return grad_x, None, None


# ---------------------------------------------------------
# Linear layer + full weight matrix
# ---------------------------------------------------------
class CompActLinear(nn.Module):

    def __init__(self, in_features, out_features, rank_scale):
        super().__init__()

        # Full weight matrix:
        # W.shape = [out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device="cuda", dtype=torch.bfloat16)
        )
        self.weight.small_grad = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.projector = GaussianProjector(in_features, out_features // rank_scale)

    def forward(self, x):
        return _CompActLinear.apply(x, self.weight, self.projector)


class FFN(nn.Module):
    def __init__(self, in_features, comp_scale):
        super().__init__()
        self.lin1 = CompActLinear(in_features, int(in_features * 5.25), comp_scale)
        self.lin2 = CompActLinear(int(in_features * 5.25), in_features, comp_scale)

        # self.W2 = torch.nn.Parameter(torch.randn(in_features, int(in_features * 5.25),device="cuda", dtype=torch.bfloat16))
        # nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    @torch.compile()
    def forward_relu(self, x):
        x = self.lin1(x)
        x.relu_()
        x = self.lin2(x)
        return x

    # @torch.compile()
    def forward_relu2(self, x):
        x = self.lin1(x)
        x.relu_()
        x = x.square()
        x = self.lin2(x)
        return x


class FFNCompAct(nn.Module):
    def __init__(self, in_features, layers, rank_scale):
        super().__init__()
        self.layers = nn.ModuleList([
            FFN(in_features, rank_scale) for _ in range(layers)
        ])

    def forward(self, x, relu2=False):
        for l in self.layers:
            x_inner = F.rms_norm(x, x.shape[1:])
            if relu2:
                x = x + l.forward_relu2(x_inner)
            else:
                x = x + l.forward_relu(x_inner)
        return x


def run_test(x, model, steps, relu2):

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")

    start = time.perf_counter()

    for _ in range(steps):
        x.grad = None
        model.zero_grad()
        torch.cuda.reset_peak_memory_stats("cuda")
        y = model.forward(x, relu2=relu2)
        loss = (y - x).abs().mean()
        del y
        loss.backward()
        loss.detach()

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps

    return avg_time, allocated


def main():
    import csv

    bs = 16_000

    ratios = [2, 4, 8]

    for r in ratios:
        print(f'{"=" * 20} {r=}')
        model = FFNCompAct(4096, 8, r).cuda().to(torch.bfloat16)
        setup_hooks(model)
        x = torch.randn(bs, 4096, device="cuda", dtype=torch.bfloat16)

        with open(f"./results/relu2_compact.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "method", "vram", "avg_time",
            ])

            for _ in range(5):
                run_test(x, model, 1, relu2=True)
                time, vram = run_test(x, model, 2, relu2=True)
                print(f'{time=}, {vram=}')
                writer.writerow([f"compact_{r}", vram, time])
                f.flush()


if __name__ == '__main__':
    main()
