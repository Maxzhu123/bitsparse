import torch
from torch import Tensor
from torch.autograd import Function
from typing import Callable

from trys import compress_tensor, decompress_tensor


class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, compressor):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        ctx.compressor = compressor

        h = z.relu_()
        h_sparse = compressor.compress_tensor(h)
        ctx.h_sparse = h_sparse
        y = h @ W.T
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        compressor = ctx.compressor
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]
        compressed, meta = ctx.h_sparse
        ctx.h_sparse = None

        h = compressor.decompress_tensor(compressed, meta)

        grad_W2 = torch.mm(grad_output.t(), h)

        # Gradients for input
        if needs_z:
            grad_h = grad_output @ W
            grad_z = grad_h * (h > 0)
        else:
            grad_z = None
        return grad_z, grad_W2, None


class FFNRelu:
    """ FFN block with relu activation"""
    @staticmethod
    def apply(x, W1, W2, compressor):
        """ FFN block with relu2 activation, 2 linear layers.
            x.shape = [*bs, d_in]
            W1.shape = [d_ff, d_in]
            W2.shape = [d_out, d_ff]

            out.shape = [*bs, d_out]
        """
        bs_dims = x.shape[:-1]          # [*bs, d_in]
        x = x.reshape(-1, x.shape[-1])  # [batch, d_in]

        z = x @ W1.T
        y = ReluLinear.apply(z, W2, compressor)

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        return y
