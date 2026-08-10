import math
import torch
import torch.nn as nn


# ---------------------------------------------------------
# Projection + gradient decoder
# ---------------------------------------------------------

class GaussianProjector(nn.Module):
    """
    R: [in_features, rank]

    encode:  x       -> x @ R
    decode:  g_small -> g_small @ R.T
    """
    def __init__(self, in_features, rank, scale=1.0, seed=0):
        super().__init__()

        if isinstance(rank, float):
            rank = round(in_features * rank)

        g = torch.Generator()
        g.manual_seed(seed)

        R = torch.randn(in_features, rank, generator=g)
        R /= math.sqrt(rank)

        self.register_buffer("R", R)
        self.scale = scale

    def project(self, x):
        return x @ self.R

    def decode(self, grad_small):
        return (grad_small @ self.R.T) * self.scale


# ---------------------------------------------------------
# Custom autograd operation
# ---------------------------------------------------------

class _CompActLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, projector):

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

        # Flatten batch / sequence dimensions:
        #
        # grad_output: [..., out]
        # x_small:     [..., rank]
        #
        # -> [out, rank]
        go = grad_output.reshape(-1, grad_output.shape[-1])
        xs = x_small.reshape(-1, x_small.shape[-1])

        grad_weight_small = go.T @ xs

        # PyTorch does not allow weight.grad to have a
        # different shape from weight, so CompAct stores it separately.
        if getattr(weight, "small_grad", None) is None:
            weight.small_grad = grad_weight_small
        else:
            weight.small_grad += grad_weight_small


        # No normal weight gradient is returned.
        return grad_x, None, None


# ---------------------------------------------------------
# Linear layer + full weight matrix
# ---------------------------------------------------------

class CompActLinear(nn.Module):

    def __init__(self, in_features, out_features, comp_scale, scale=1.0):
        super().__init__()

        # Full weight matrix:
        # W.shape = [out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.projector = GaussianProjector(
            in_features,
            out_features // comp_scale,
            scale=scale,
        )

        self.weight.small_grad = None

    def forward(self, x):
        return _CompActLinear.apply(
            x,
            self.weight,
            self.projector,
        )


# ---------------------------------------------------------
# Minimal optimizer step
# ---------------------------------------------------------

@torch.no_grad()
def compact_sgd_step(layer, lr):

    # Decode [out, rank] -> [out, in]
    if layer.weight.small_grad is not None:

        grad_weight = layer.projector.decode(
            layer.weight.small_grad
        )

        layer.weight.add_(grad_weight, alpha=-lr)

        layer.weight.small_grad = None


layer = CompActLinear(
    in_features=4096,
    out_features=21504,
    comp_scale=2,
)

x = torch.randn(2, 128, 4096)

y = layer(x)
loss = y.square().mean()
loss.backward()

print(layer.weight.shape)

print(layer.weight.small_grad.shape)

compact_sgd_step(layer, lr=1e-3)
