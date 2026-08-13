import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F

from .config import CACHE_FP8_MATMUl

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


def is_fp8(dtype) -> bool:
    """True for the supported 8-bit float storage dtypes (e4m3fn, e5m2)."""
    return dtype in _FP8_DTYPES


class MatmulFp8(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, a_scale: Tensor|None=None, b_scale: Tensor|None=None) -> Tensor:
        if not CACHE_FP8_MATMUl:
            ctx.save_for_backward(a, b)
            ctx.a_scale, ctx.b_scale = a_scale, b_scale

        # Convert to fp8 if necessary
        if a.dtype == torch.bfloat16:
            assert a_scale is None
            a_fp8, a_scale = to_fp8(a)
        else:
            assert a_scale is not None
            a_fp8 = a
        if b.dtype == torch.bfloat16:
            assert b_scale is None
            b_fp8, b_scale = to_fp8(b)
        else:
            assert b_scale is not None
            b_fp8 = b

        if CACHE_FP8_MATMUl:
            ctx.save_for_backward(a_fp8, b_fp8, a_scale, b_scale)

        out = F.scaled_mm(
            mat_a=a_fp8, mat_b=b_fp8,
            scale_a=a_scale, scale_b=b_scale, output_dtype=torch.bfloat16,
            scale_recipe_a=F.ScalingType.TensorWise, scale_recipe_b=F.ScalingType.TensorWise
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if CACHE_FP8_MATMUl:
            a_fp8, b_fp8, a_scale, b_scale = ctx.saved_tensors
        else:
            a, b = ctx.saved_tensors
            a_scale, b_scale = ctx.a_scale, ctx.b_scale
            if a.dtype == torch.bfloat16:
                assert a_scale is None
                a_fp8, a_scale = to_fp8(a)
            else:
                assert a_scale is not None
                a_fp8 = a
            if b.dtype == torch.bfloat16:
                assert b_scale is None
                b_fp8, b_scale = to_fp8(b)
            else:
                assert b_scale is not None
                b_fp8 = b

        grad_out_fp8, grad_out_scale = to_fp8(grad_output)
        grad_a, grad_b = None, None
        if ctx.needs_input_grad[0]:
            grad_a = F.scaled_mm(
                grad_out_fp8, b_fp8.T,
                scale_a=grad_out_scale, scale_b=b_scale, output_dtype=a_fp8.dtype,
                scale_recipe_a=F.ScalingType.TensorWise, scale_recipe_b=F.ScalingType.TensorWise
            )
        if ctx.needs_input_grad[1]:
            grad_b = F.scaled_mm(
                a_fp8.T.contiguous(), grad_out_fp8,
                scale_a=a_scale, scale_b=grad_out_scale, output_dtype=b_fp8.dtype,
                scale_recipe_a=F.ScalingType.TensorWise, scale_recipe_b=F.ScalingType.TensorWise
            )

        return grad_a, grad_b, None, None


def matmul(a: Tensor, b: Tensor, fp8: bool, a_scale:Tensor|None=None, b_scale:Tensor|None=None) -> Tensor:
    """ Compute a @ b. Supports fp8 inputs. """
    return MatmulFp8.apply(a, b, a_scale=a_scale, b_scale=b_scale) if fp8 else a @ b


@torch.no_grad()
@torch.compile()
def to_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    scale = (x.detach().abs().max() / _FP8_MAX).to(torch.float32)
    scale = scale.clamp(min=1e-9)
    x_fp8 = (x / scale).to(_FP8_DTYPE).contiguous()
    return x_fp8, scale
