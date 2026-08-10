import math
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import gc
import time


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
    """
    def __init__(self, in_features, rank):
        super().__init__()

        self.in_features = in_features
        self.rank = rank

    @torch.compiler.disable()
    def project(self, x):
        self.seed = random.randint(0, 2**16)
        g = torch.Generator(device="cuda")
        g.manual_seed(self.seed)

        R = torch.randn(self.in_features, self.rank, generator=g, device="cuda", dtype=torch.bfloat16)
        R /= math.sqrt(self.rank)

        return x @ R

    @torch.compiler.disable()
    def decode(self, grad_small):
        g = torch.Generator(device="cuda")
        g.manual_seed(self.seed)

        R = torch.randn(self.in_features, self.rank, generator=g, device="cuda", dtype=torch.bfloat16)
        R /= math.sqrt(self.rank)

        return grad_small @ R.T


# ---------------------------------------------------------
# Custom autograd operation
# ---------------------------------------------------------
class _CompActLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, projector):
        weight.projector = projector

        # Exact forward pass using the full weight matrix.
        y = x @ weight.T

        # The important part:
        # save compressed x rather than full x.
        x_small = projector.project(x)

        ctx.save_for_backward(x_small, weight)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_small, weight = ctx.saved_tensors

        # Gradient w.r.t. activation is still exact.
        grad_x = grad_output @ weight

        grad_weight_small = grad_output.T @ x_small

        weight.small_grad = grad_weight_small
        return grad_x, None , None


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
        # self.lin2 = CompActLinear(int(in_features * 5.25), in_features, comp_scale)

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
        # x = ReLUSquared.apply(x)
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
    # torch._dynamo.config.allow_unspec_int_on_nn_module = True

    bs = 16_000

    model = FFNCompAct(4096, 8, 2).cuda().to(torch.bfloat16)
    setup_hooks(model)
    x = torch.randn(bs, 4096, device="cuda", dtype=torch.bfloat16)

    run_test(x, model, 1, relu2=True)
    time, vram = run_test(x, model, 5, relu2=True)
    print(f'{time=}, {vram=}')


if __name__ == '__main__':
    main()
