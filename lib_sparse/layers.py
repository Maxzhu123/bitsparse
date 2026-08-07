from typing import TYPE_CHECKING
import torch
from torch import Tensor
from torch.autograd import Function

from .code.functions import dense_to_tilesparse
from .code.sparse_matmul import AspB, AspRelu2B
from .code.triton_operators import mask_with_bitmask_, relu2_grad_sparse_
from .bitsparse import BitsparseTensor, RELU2_SCALE
if TYPE_CHECKING:
    from bitsparse import TensorBuffer

# ------------------------------------------------------------
# ReLU layers
# ------------------------------------------------------------
class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data:TensorBuffer|None=None, packed_15bit: bool=False):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = dense_to_tilesparse(h, sparse_data, packed_15bit)
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
        return grad_z, grad_W2, None, None


class FFNRelu:
    """ FFN block with relu activation"""
    @staticmethod
    def apply(x, W1, W2, b1=None, b2=None,
              sparse_data:TensorBuffer|None=None, packed_15bit: bool=False):
        """ FFN block with relu2 activation, 2 linear layers.
            x.shape = [*bs, d_in]
            W1.shape = [d_ff, d_in]
            W2.shape = [d_ff, d_out]
            b1.shape = [d_ff]
            b2.shape = [d_out]

            out.shape = [*bs, d_out]
        """
        bs_dims = x.shape[:-1]          # [*bs, d_in]
        x = x.reshape(-1, x.shape[-1])  # [batch, d_in]

        z = x @ W1.T
        if b1 is not None:
            z = z + b1
        y = ReluLinear.apply(z, W2, sparse_data, packed_15bit)

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        if b2 is not None:
            y = y + b2

        return y


class FFNRelu_3:
    """ FFN block with relu activation, 3 linear layers. """
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data:TensorBuffer|None=None, packed_15bit: bool=False):
        z1 = x @ W1.T
        y1 = ReluLinear.apply(z1, W2, sparse_data, packed_15bit)
        y2 = ReluLinear.apply(y1, W3, sparse_data, packed_15bit)
        return y2

# ------------------------------------------------------------
# ReLU2 layers
# ------------------------------------------------------------
class Relu2Linear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data:TensorBuffer|None, packed_15bit: bool=False):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = dense_to_tilesparse(h, sparse_data, packed_15bit)
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

        return grad_z, grad_W2, None, None


class FFNRelu2:
    @staticmethod
    def apply(x, W1, W2, b1=None, b2=None,
              sparse_data:TensorBuffer|None=None, packed_15bit: bool=False):
        """ FFN block with relu2 activation, 2 linear layers.
            x.shape = [*bs, d_in]
            W1.shape = [d_ff, d_in]
            W2.shape = [d_ff, d_out]
            b1.shape = [d_ff]
            b2.shape = [d_out]

            out.shape = [*bs, d_out]
        """
        bs_dims = x.shape[:-1]          # [*bs, d_in]
        x = x.reshape(-1, x.shape[-1])  # [batch, d_in]
        z = x @ W1.T                    # [batch, d_ff]
        if b1 is not None:
            z = z + b1
        y = Relu2Linear.apply(z, W2, sparse_data, packed_15bit) # [batch, d_out]
        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        if b2 is not None:
            y = y + b2
        return y


class FFNRelu2_3:
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data:TensorBuffer|None=None, packed_15bit: bool=False):
        z1 = x @ W1.T
        y1 = Relu2Linear.apply(z1, W2, sparse_data, packed_15bit)
        y2 = Relu2Linear.apply(y1, W3, sparse_data, packed_15bit)
        return y2

# ------------------------------------------------------------
# Manual implemented layers
# ------------------------------------------------------------
class FFNSparse(Function):
    """Forward of FFN."""

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data:TensorBuffer|None=None, packed_15bit: bool=False):
        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h, sparse_data, packed_15bit)
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
        return grad_x, grad_W1, grad_W2, None, None


class FFNSparseRelu2(Function):
    """Autograd FFN using sparse storage for ReLU-squared hidden activation.
    Formula:
        z = x @ W1.T
        h = k * relu(z^2)
        out = z @ W2.T
        k = 1 / sqrt(3) matches the RMS of ReLU for standard-normal inputs.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data: TensorBuffer | None=None, packed_15bit: bool=False):
        bs_dims = x.shape[:-1]          # [*bs, d]
        x = x.reshape(-1, x.shape[-1])

        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h, sparse_data, packed_15bit)
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

        grad_W2 = AspRelu2B(grad_output.T, h)

        grad_h2 = grad_output @ W2
        grad_z = relu2_grad_sparse_(grad_h2, h)
        del h

        if needs_x:
            grad_x = grad_z @ W1
            grad_x = grad_x.reshape(*bs_dims, -1)
        else:
            grad_x = None
        grad_W1 = grad_z.T @ x
        return grad_x, grad_W1, grad_W2, None, None

