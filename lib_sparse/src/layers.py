from typing import TYPE_CHECKING
import torch
from torch import Tensor
from torch.autograd import Function

from src.code.functions import dense_to_tilesparse
from src.code.sparse_matmul import AspB, AspRelu2B
from src.code.triton_operators import mask_with_bitmask_, relu2_grad_sparse_
from src.bitsparse import BitsparseTensor, RELU2_SCALE
if TYPE_CHECKING:
    from src.bitsparse import TensorBuffer

# ------------------------------------------------------------
# ReLU layers
# ------------------------------------------------------------
class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data:TensorBuffer|None=None):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = dense_to_tilesparse(h, sparse_data)
        ctx.h_sparse = h_sparse
        y = h @ W.T
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]
        h: BitsparseTensor = ctx.h_sparse
        ctx.h_sparse = None

        grad_W2 = AspB(grad_output.T, h)

        # Gradients for input
        if needs_z:
            grad_h = grad_output @ W
            grad_z = mask_with_bitmask_(grad_h, h)
        else:
            grad_z = None
        return grad_z, grad_W2, None


class FFNRelu:
    """ FFN block with relu activation"""
    @staticmethod
    def apply(x, W1, W2, sparse_data:TensorBuffer|None=None):
        z = x @ W1.T
        y = ReluLinear.apply(z, W2, sparse_data)

        return y


class FFNRelu_3:
    """ FFN block with relu activation, 3 linear layers. """
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data:TensorBuffer|None=None):
        z1 = x @ W1.T
        y1 = ReluLinear.apply(z1, W2, sparse_data)
        y2 = ReluLinear.apply(y1, W3, sparse_data)
        return y2

# ------------------------------------------------------------
# ReLU2 layers
# ------------------------------------------------------------
class Relu2Linear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data:TensorBuffer|None):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = dense_to_tilesparse(h, sparse_data)
        ctx.h_sparse = h_sparse
        h.square_()
        h.mul_(RELU2_SCALE)
        y = h @ W.T
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]
        h: BitsparseTensor = ctx.h_sparse
        ctx.h_sparse = None

        grad_W2 = AspRelu2B(grad_output.T, h)

        # Needs gradient for z
        if needs_z:
            grad_h = grad_output @ W
            grad_z = relu2_grad_sparse_(grad_h, h)
        else:
            grad_z = None

        return grad_z, grad_W2, None


class FFNRelu2:
    @staticmethod
    def apply(x, W1, W2, sparse_data:TensorBuffer|None=None):
        """ FFN block with relu2 activation, 2 linear layers.
            x.shape = [*bs, d]
            W1.shape = [d2, d]
            W2.shape = [d3, d]
            out.shape = [*bs, d3]
        """
        bs_dims = x.shape[:-1]          # [*bs, d]
        x = x.reshape(-1, x.shape[-1])
        z = x @ W1.T
        y = Relu2Linear.apply(z, W2, sparse_data)
        y = y.reshape(*bs_dims,  y.shape[-1])
        return y


class FFNRelu2_3:
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data:TensorBuffer|None=None):
        z1 = x @ W1.T
        y1 = Relu2Linear.apply(z1, W2, sparse_data)
        y2 = Relu2Linear.apply(y1, W3, sparse_data)
        return y2

# ------------------------------------------------------------
# Manual implemented layers
# ------------------------------------------------------------
class FFNSparse(Function):
    """Forward of FFN."""

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data:TensorBuffer|None=None):
        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h, sparse_data)
        return h @ W2.T

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Compute FFN gradients."""
        x, W1, W2 = ctx.saved_tensors
        h: BitsparseTensor = ctx.h_sparse
        ctx.h_sparse = None
        needs_x = ctx.needs_input_grad[0]

        grad_W2 = AspB(grad_output.T, h)

        grad_h = grad_output @ W2
        grad_z = mask_with_bitmask_(grad_h, h)
        del h

        if needs_x:
            grad_x = grad_z @ W1
        else:
            grad_x = None

        grad_W1 = grad_z.T @ x
        return grad_x, grad_W1, grad_W2, None


class FFNSparseRelu2(Function):
    """Autograd FFN using sparse storage for ReLU-squared hidden activation.
    Formula:
        z = x @ W1.T
        h = k * relu(z^2)
        out = z @ W2.T
        k = 1 / sqrt(3) matches the RMS of ReLU for standard-normal inputs.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data: TensorBuffer | None=None):
        bs_dims = x.shape[:-1]          # [*bs, d]
        x = x.reshape(-1, x.shape[-1])

        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h, sparse_data)
        h.square_()
        h.mul_(RELU2_SCALE)
        out = h @ W2.T
        out = out.reshape(*bs_dims, -1)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``.
            grad_output.shape = [*bs, in_dim]
        """
        bs_dims = grad_output.shape[:-1]  # [*bs, in_dim]
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        x, W1, W2 = ctx.saved_tensors
        h = ctx.h_sparse
        ctx.h_sparse = None
        needs_x = ctx.needs_input_grad[0]

        grad_W2 = AspRelu2B(grad_output.T, h)  # AspRelu2B_block(grad_output.T, z) #

        grad_h2 = grad_output @ W2
        grad_z = relu2_grad_sparse_(grad_h2, h)
        del h

        if needs_x:
            grad_x = grad_z @ W1
            grad_x = grad_x.reshape(*bs_dims, -1)
        else:
            grad_x = None
        grad_W1 = grad_z.T @ x
        return grad_x, grad_W1, grad_W2, None



