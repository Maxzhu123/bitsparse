import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F

from .config import CACHE_FP8_MATMUl

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max

# Blackwell (cc >= 12) accepts row-major ``mat_b`` directly.  Ada (cc < 12)
# requires the cuBLASLt layout (A row-major, B column-major) or scaled_mm
# raises CUBLAS_STATUS_NOT_SUPPORTED.
_NEEDS_LAYOUT_REFORMAT: bool = (
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 12
)


def is_fp8(dtype) -> bool:
    """True for the supported 8-bit float storage dtypes (e4m3fn, e5m2)."""
    return dtype in _FP8_DTYPES


def _scaled_mm(a: Tensor, b: Tensor, a_scale: Tensor, b_scale: Tensor, output_dtype: torch.dtype) -> Tensor:
    """``F.scaled_mm`` with the operand layout cuBLASLt requires for FP8.

    ``mat_a`` must always be row-major contiguous.  On Ada (RTX 40-series,
    cc < 12) cuBLASLt additionally wants B column-major; other layouts raise
    ``CUBLAS_STATUS_NOT_SUPPORTED``.  Blackwell (RTX 50-series, cc >= 12)
    accepts the row-major B directly, so the reformat is skipped there to
    avoid the extra copies.
    """
    # cuBLASLt FP8 wants A row-major contiguous (all architectures).
    # No allocation if already row-major contiguous.
    if a.stride(1) != 1:
        a = a.contiguous()

    if _NEEDS_LAYOUT_REFORMAT:
        # cuBLASLt FP8 wants B column-major (Ada only).
        # Convert while preserving B's logical [K, N] shape.
        if b.stride(0) != 1:
            b = b.T.contiguous().T

    return F.scaled_mm(
        mat_a=a, mat_b=b,
        scale_a=a_scale, scale_b=b_scale, output_dtype=output_dtype,
        scale_recipe_a=F.ScalingType.TensorWise, scale_recipe_b=F.ScalingType.TensorWise,
    )


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

        out = _scaled_mm(
            a_fp8.contiguous(), b_fp8,
            a_scale, b_scale, output_dtype=torch.bfloat16,
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
            grad_a = _scaled_mm(
                grad_out_fp8, b_fp8.T,
                grad_out_scale, b_scale, output_dtype=a_fp8.dtype,
            )
        if ctx.needs_input_grad[1]:
            grad_b = _scaled_mm(
                a_fp8.T.contiguous(), grad_out_fp8,
                a_scale, grad_out_scale, output_dtype=b_fp8.dtype,
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
    x_fp8 = (x / scale).to(_FP8_DTYPE)
    return x_fp8, scale
