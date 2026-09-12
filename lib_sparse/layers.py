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


# ------------------------------------------------------------
# Fused RMS norm-linear
# ------------------------------------------------------------
class FusedRMSNormMLP(Function):
    """Bias-free RMSNorm/linear with BF16 inputs/weights and optional FP8 GEMMs."""

    @staticmethod
    def forward(ctx, x: Tensor, W1: Tensor, norm_weight: Tensor | None = None,
                eps: float = 1e-6, fp8: bool = False):
        """BF16 x[..., in] and W1[out, in]; fp8 enables scaled E4M3 GEMMs."""
        normalized, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], norm_weight, eps,
        )
        ctx.save_for_backward(x, W1, norm_weight, rstd)
        ctx.eps = eps
        ctx.fp8 = fp8
        output = matmul(normalized.reshape(-1, x.shape[-1]), W1.T, fp8)
        return output.reshape(*x.shape[:-1], W1.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: Tensor):
        x, W1, norm_weight, rstd = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, W1.shape[0])
        if ctx.fp8:
            grad_output, grad_scale = to_fp8(grad_output)
        else:
            grad_scale = None
        grad_normalized = matmul(grad_output, W1, ctx.fp8, a_scale=grad_scale).reshape(x.shape)
        grad_x, grad_norm_weight = torch.ops.aten._fused_rms_norm_backward.default(
            grad_normalized, x, [x.shape[-1]], rstd, norm_weight,
            [True, norm_weight is not None],
        )
        del grad_normalized

        # Recompute the BF16 projection input and allocate the large weight gradient last.
        normalized = torch.nn.functional.rms_norm(x, [x.shape[-1]], norm_weight, ctx.eps)
        grad_W1 = matmul(grad_output.T, normalized.reshape(-1, x.shape[-1]), ctx.fp8, a_scale=grad_scale)
        return grad_x, grad_W1, grad_norm_weight, None, None


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
            grad_output, scale = to_fp8(grad_output)
        else:
            grad_output, scale = grad_output, None

        grad_W2 = AspB(grad_output.T, h, A_scale=scale)

        # Gradients for input
        if needs_z:
            grad_h = matmul(grad_output, W, fp8=fp8, a_scale=scale)
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


class RMSFFNRelu:
    """Sparse ReLU FFN: BF16 inputs/weights, BF16 or FP8 compute/storage."""

    @staticmethod
    def apply(x: Tensor, W1: Tensor, W2: Tensor, norm_weight: Tensor | None = None, *,
              eps: float = 1e-6, sparse_data: TensorBuffer | None = None,
              pack_sbit: bool = False, storage_dtype: torch.dtype = torch.bfloat16):
        """x[..., in], optional norm_weight[in], W1[ff, in], W2[out, ff]."""
        batch_dims = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        z = FusedRMSNormMLP.apply(x, W1, norm_weight, eps, is_fp8(storage_dtype))
        y = ReluLinear.apply(z, W2, sparse_data, pack_sbit, storage_dtype)
        return y.reshape(*batch_dims, y.shape[-1])


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

        # Use fp8 grad_output
        fp8 = is_fp8(ctx.dtype)
        if fp8:
            grad_output, scale = to_fp8(grad_output)
        else:
            grad_output, scale = grad_output, None

        grad_W2 = AspRelu2B(grad_output.T, h, A_scale=scale)

        # Needs gradient for z
        if needs_z:
            grad_h = matmul(grad_output, W, fp8=fp8, a_scale=scale)
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


class RMSFFNRelu2:
    """Sparse ReLU² FFN: BF16 inputs/weights, BF16 or FP8 compute/storage."""

    @staticmethod
    def apply(x: Tensor, W1: Tensor, W2: Tensor, norm_weight: Tensor=None, *,
              eps: float = 1e-6, sparse_data: TensorBuffer | None = None,
              pack_sbit: bool = False, storage_dtype: torch.dtype = torch.bfloat16):
        """x[..., in], norm_weight[in], W1[ff, in], W2[out, ff]."""
        batch_dims = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        z = FusedRMSNormMLP.apply(x, W1, norm_weight, eps, is_fp8(storage_dtype))
        y = Relu2Linear.apply(z, W2, sparse_data, pack_sbit, storage_dtype)
        return y.reshape(*batch_dims, y.shape[-1])
