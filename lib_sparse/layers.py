from typing import TYPE_CHECKING
import torch
from torch import Tensor
from torch.autograd import Function

from .src.functions import dense_to_tilesparse
from .src.sparse_matmul import AspB, AspRelu2B
from .src.triton_operators import mask_with_bitmask_, relu2_grad_sparse_
from .bitsparse import BitsparseTensor
from .fp8 import is_fp8, matmul, to_fp8
from .config import RELU2_SCALE

if TYPE_CHECKING:
    from bitsparse import TensorBuffer


def print_memory(msg, max=True):
    from cprint import c_print
    if max:
        memory = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    else:
        memory = torch.cuda.memory_allocated("cuda") / 1024 ** 2
    c_print(f'{msg}: {memory:.2f} MB', color="bright_cyan")


# ------------------------------------------------------------
# ReLU layers
# ------------------------------------------------------------
class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data:TensorBuffer|None=None, pack_sbit: bool=False,
                dtype: torch.dtype=torch.bfloat16):
        """ relu(Wx) layer. """
        ctx.dtype = dtype
        ctx.save_for_backward(W)
        h = z.relu_()

        # Quantize input if needed
        if is_fp8(dtype):
            h, scale = to_fp8(h)
        else:
            scale = None

        h_sparse = dense_to_tilesparse(h, scale, sparse_data, pack_sbit)

        ctx.h_sparse = h_sparse
        y = matmul(h, W.T, is_fp8(dtype), a_scale=scale)
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]
        h: BitsparseTensor = ctx.h_sparse
        ctx.h_sparse = None

        fp8 = is_fp8(ctx.dtype)
        if fp8:
            grad_output_fp8, scale = to_fp8(grad_output)
            del grad_output
        else:
            grad_output_fp8, scale = grad_output, None

        grad_W2 = AspB(grad_output_fp8.T.contiguous(), h, A_scale=scale)

        # Gradients for input
        if needs_z:
            grad_h = matmul(grad_output_fp8, W, fp8=fp8, a_scale=scale)
            grad_z = mask_with_bitmask_(grad_h, h)
        else:
            grad_z = None
        return grad_z, grad_W2, None, None, None


class FFNRelu:
    """ FFN block with relu activation"""
    @staticmethod
    def apply(x, W1, W2, b1=None, b2=None, *,
              sparse_data:TensorBuffer|None=None, pack_sbit: bool=False, dtype: torch.dtype=torch.bfloat16):
        """ FFN block with relu activation, 2 linear layers.
            x.shape = [*bs, d_in]
            W1.shape = [d_ff, d_in]
            W2.shape = [d_ff, d_out]
            b1.shape = [d_ff]
            b2.shape = [d_out]

            out.shape = [*bs, d_out]
        """
        bs_dims = x.shape[:-1]          # [*bs, d_in]
        x = x.reshape(-1, x.shape[-1])  # [batch, d_in]

        z = matmul(x, W1.T, is_fp8(dtype))
        if b1 is not None:
            z = z + b1
        y = ReluLinear.apply(z, W2, sparse_data, pack_sbit, dtype)

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        if b2 is not None:
            y = y + b2
        return y


class FFNRelu_3:
    """ FFN block with relu activation, 3 linear layers. """
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data:TensorBuffer|None=None, pack_sbit: bool=False,
              storage_dtype: torch.dtype = torch.bfloat16):
        z1 = matmul(x, W1.T, is_fp8(storage_dtype))
        y1 = ReluLinear.apply(z1, W2, sparse_data, pack_sbit, storage_dtype)
        y2 = ReluLinear.apply(y1, W3, sparse_data, pack_sbit, storage_dtype)
        return y2

# ------------------------------------------------------------
# ReLU2 layers
# ------------------------------------------------------------
class Relu2Linear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data:TensorBuffer|None, pack_sbit: bool=False,
                storage_dtype: torch.dtype = torch.bfloat16):
        """ relu(Wx) layer. """
        ctx.dtype = storage_dtype
        ctx.save_for_backward(W)
        h = z.relu_()

        # Quantize input if needed
        if is_fp8(storage_dtype):
            h_stored, scale = to_fp8(h)
        else:
            h_stored, scale = h, None

        h_sparse = dense_to_tilesparse(h_stored, scale, sparse_data, pack_sbit)
        ctx.h_sparse = h_sparse

        # Forward matmul uses the squared activation
        h.square_()
        h.mul_(RELU2_SCALE)
        y = matmul(h, W.T, is_fp8(storage_dtype))
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]
        h: BitsparseTensor = ctx.h_sparse
        ctx.h_sparse = None

        fp8 = is_fp8(ctx.dtype)
        if fp8:
            grad_output_fp8, scale = to_fp8(grad_output)
            del grad_output
        else:
            grad_output_fp8, scale = grad_output, None

        grad_W2 = AspRelu2B(grad_output_fp8.T.contiguous(), h, A_scale=scale)

        # Needs gradient for z
        if needs_z:
            grad_h = matmul(grad_output_fp8, W, fp8=fp8, a_scale=scale)
            grad_z = relu2_grad_sparse_(grad_h, h)
        else:
            grad_z = None

        return grad_z, grad_W2, None, None, None


class FFNRelu2:
    @staticmethod
    def apply(x, W1, W2, b1=None, b2=None, *,
              sparse_data:TensorBuffer|None=None, pack_sbit: bool=False,
              storage_dtype: torch.dtype = torch.bfloat16):
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
        z = matmul(x, W1.T, is_fp8(storage_dtype)) # [batch, d_ff]
        if b1 is not None:
            z = z + b1
        y = Relu2Linear.apply(z, W2, sparse_data, pack_sbit, storage_dtype) # [batch, d_out]
        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        if b2 is not None:
            y = y + b2
        return y


class FFNRelu2_3:
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data:TensorBuffer|None=None, pack_sbit: bool=False,
              storage_dtype: torch.dtype = torch.bfloat16):
        z1 = matmul(x, W1.T, is_fp8(storage_dtype))
        y1 = Relu2Linear.apply(z1, W2, sparse_data, pack_sbit, storage_dtype)
        y2 = Relu2Linear.apply(y1, W3, sparse_data, pack_sbit, storage_dtype)
        return y2

# # ------------------------------------------------------------
# # Manual implemented layers
# # ------------------------------------------------------------
# class FFNSparse(Function):
#     """Forward of FFN."""
#
#     @staticmethod
#     def forward(ctx, x, W1, W2, sparse_data:TensorBuffer|None=None, pack_sbit: bool=False,
#                 storage_dtype: torch.dtype = torch.bfloat16):
#         ctx.save_for_backward(x, W1, W2)
#         z = matmul(x, W1.T, is_fp8(storage_dtype))
#         h = z.relu_()
#         ctx.h_sparse = dense_to_tilesparse(h, sparse_data, pack_sbit, storage_dtype)
#         return matmul(h, W2.T, is_fp8(storage_dtype))
#
#     @staticmethod
#     def backward(ctx, grad_output: Tensor):
#         """Compute FFN gradients."""
#         x, W1, W2 = ctx.saved_tensors
#         h: BitsparseTensor = ctx.h_sparse
#         ctx.h_sparse = None
#         needs_x = ctx.needs_input_grad[0]
#
#         grad_W2 = AspB(grad_output.T, h)
#
#         grad_h = grad_output @ W2
#         grad_z = mask_with_bitmask_(grad_h, h)
#         del h
#
#         if needs_x:
#             grad_x = grad_z @ W1
#         else:
#             grad_x = None
#
#         grad_W1 = grad_z.T @ x
#         return grad_x, grad_W1, grad_W2, None, None, None
#
#
# class FFNSparseRelu2(Function):
#     """Autograd FFN using sparse storage for ReLU-squared hidden activation.
#     Formula:
#         z = x @ W1.T
#         h = k * relu(z^2)
#         out = z @ W2.T
#         k = 1 / sqrt(3) matches the RMS of ReLU for standard-normal inputs.
#     """
#     @staticmethod
#     def forward(ctx, x, W1, W2, sparse_data: TensorBuffer|None=None, pack_sbit: bool=False,
#                 storage_dtype: torch.dtype = torch.bfloat16):
#         bs_dims = x.shape[:-1]          # [*bs, d]
#         x = x.reshape(-1, x.shape[-1])
#
#         ctx.save_for_backward(x, W1, W2)
#         z = _mm(x, W1.T, storage_dtype)
#         h = z.relu_()
#         ctx.h_sparse = dense_to_tilesparse(h, sparse_data, pack_sbit, storage_dtype)
#         h.square_()
#         h.mul_(RELU2_SCALE)
#         out = _mm(h, W2.T, storage_dtype)
#         out = out.reshape(*bs_dims, -1)
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output: Tensor):
#         """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``.
#             grad_output.shape = [*bs, in_dim]
#         """
#         bs_dims = grad_output.shape[:-1]  # [*bs, in_dim]
#         grad_output = grad_output.reshape(-1, grad_output.shape[-1])
#         x, W1, W2 = ctx.saved_tensors
#         h = ctx.h_sparse
#         ctx.h_sparse = None
#         needs_x = ctx.needs_input_grad[0]
#
#         grad_W2 = AspRelu2B(grad_output.T, h)
#
#         grad_h2 = grad_output @ W2
#         grad_z = relu2_grad_sparse_(grad_h2, h)
#         del h
#
#         if needs_x:
#             grad_x = grad_z @ W1
#             grad_x = grad_x.reshape(*bs_dims, -1)
#         else:
#             grad_x = None
#         grad_W1 = grad_z.T @ x
#         return grad_x, grad_W1, grad_W2, None, None, None
