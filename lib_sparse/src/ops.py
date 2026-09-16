"""Opaque RMSNorm/linear and standalone sparse-conversion operators.

Capturing variable-sized storage requires callers to enable
torch._dynamo.config.capture_dynamic_output_shape_ops.
"""
import torch
from torch import Tensor

from .functions import dense_to_tilesparse as _dense_to_tilesparse
from ..bitsparse import BitsparseTensor, TensorBuffer, tile_grid
from ..config import BLOCK_M, BLOCK_N
from ..fp8 import matmul, to_fp8


@torch.library.custom_op("bitsparse::rms_norm_linear", mutates_args=(), device_types="cuda")
def rms_norm_linear(
    x: Tensor, weight: Tensor, norm_weight: Tensor | None = None,
    eps: float | None = 1e-6, fp8: bool = False,
) -> tuple[Tensor, Tensor]:
    normalized, rstd = torch.ops.aten._fused_rms_norm.default(x, [x.shape[-1]], norm_weight, eps)
    output = matmul(normalized.reshape(-1, x.shape[-1]), weight.T, fp8)
    return output.reshape(*x.shape[:-1], weight.shape[0]), rstd


@rms_norm_linear.register_fake
def _rms_norm_linear_fake(x, weight, norm_weight=None, eps=1e-6, fp8=False):
    # Use native metadata inference for the statistics' shape, stride and dtype.
    _, rstd = torch.ops.aten._fused_rms_norm.default(x, [x.shape[-1]], norm_weight, eps)
    return x.new_empty((*x.shape[:-1], weight.shape[0])), rstd


@torch.library.custom_op(
    "bitsparse::rms_norm_linear_backward", mutates_args=(), device_types="cuda",
    schema="(Tensor grad_output, Tensor x, Tensor weight, Tensor? norm_weight, "
           "Tensor rstd, float? eps, bool fp8) -> (Tensor, Tensor, Tensor?)",
)
def rms_norm_linear_backward(
    grad_output: Tensor, x: Tensor, weight: Tensor, norm_weight: Tensor | None,
    rstd: Tensor, eps: float | None, fp8: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    grad_output = grad_output.reshape(-1, weight.shape[0])
    if fp8:
        grad_output, grad_scale = to_fp8(grad_output)
    else:
        grad_scale = None
    grad_normalized = matmul(grad_output, weight, fp8, a_scale=grad_scale).reshape(x.shape)
    grad_x, grad_norm_weight = torch.ops.aten._fused_rms_norm_backward.default(
        grad_normalized, x, [x.shape[-1]], rstd, norm_weight, [True, norm_weight is not None],
    )
    del grad_normalized
    # This recomputation stays inside the opaque backward: the compiler cannot
    # move it to forward and retain the normalized activation instead.
    normalized = torch.nn.functional.rms_norm(x, [x.shape[-1]], norm_weight, eps)
    grad_weight = matmul(grad_output.T, normalized.reshape(-1, x.shape[-1]), fp8, a_scale=grad_scale)
    return grad_x, grad_weight, grad_norm_weight


@rms_norm_linear_backward.register_fake
def _rms_norm_linear_backward_fake(grad_output, x, weight, norm_weight, rstd, eps, fp8):
    grad_x, grad_norm_weight = torch.ops.aten._fused_rms_norm_backward.default(
        x.new_empty(x.shape), x, [x.shape[-1]], rstd, norm_weight, [True, norm_weight is not None],
    )
    return grad_x, weight.new_empty(weight.shape), grad_norm_weight


def _rms_norm_linear_setup_context(ctx, inputs, output):
    x, weight, norm_weight, eps, fp8 = inputs
    _, rstd = output
    ctx.save_for_backward(x, weight, norm_weight, rstd)
    ctx.mark_non_differentiable(rstd)
    ctx.eps, ctx.fp8 = eps, fp8


@torch.autograd.function.once_differentiable
def _rms_norm_linear_autograd(ctx, grad_output, grad_rstd):
    x, weight, norm_weight, rstd = ctx.saved_tensors
    grad_x, grad_weight, grad_norm_weight = rms_norm_linear_backward(
        grad_output, x, weight, norm_weight, rstd, ctx.eps, ctx.fp8,
    )
    return grad_x, grad_weight, grad_norm_weight, None, None


rms_norm_linear.register_autograd(_rms_norm_linear_autograd, setup_context=_rms_norm_linear_setup_context)


@torch.library.custom_op(
    "bitsparse::dense_to_tilesparse",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.dynamic_output_shape, torch.Tag.cudagraph_unsafe),
)
def dense_to_tilesparse_op(
    dense: Tensor, pack_sbit: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Keep the host sync and exact-sized allocation inside the opaque op.
    sparse = _dense_to_tilesparse(dense, None, None, pack_sbit)
    return sparse.vals, sparse.bitmask, sparse.prefix, sparse.vals_offset


@dense_to_tilesparse_op.register_fake
def _fake(dense, pack_sbit=False):
    # Bind the symbolic size directly to the returned storage length. For packed
    # values it counts bytes (including padding/guard), rather than logical nnz.
    size = torch.library.get_ctx().new_dynamic_size(min=4 if pack_sbit else 0)
    dtype = torch.uint8 if pack_sbit else dense.dtype
    _, _, tiles, _, tile_bytes = tile_grid(*dense.shape, BLOCK_M, BLOCK_N)
    return (
        dense.new_empty((size,), dtype=dtype),
        dense.new_empty((tiles * tile_bytes,), dtype=torch.uint8),
        dense.new_empty((tiles + 1,), dtype=torch.uint32),
        dense.new_empty((), dtype=torch.int64),
    )


def dense_to_tilesparse_custom(
    dense: Tensor, scale: Tensor | None,
    sparse_data: TensorBuffer | None = None, pack_sbit: bool = False,
) -> BitsparseTensor:
    # Shared-buffer mutation is deliberately left on the original path for now.
    if sparse_data is not None:
        return _dense_to_tilesparse(dense, scale, sparse_data, pack_sbit)
    vals, bitmask, prefix, offset = dense_to_tilesparse_op(dense, pack_sbit)
    grid_m, grid_n, _, _, _ = tile_grid(*dense.shape, BLOCK_M, BLOCK_N)
    return BitsparseTensor(
        vals, bitmask, prefix, dense.shape, dense.dtype,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        scale=scale, vals_offset=offset, pack_sbit=pack_sbit,
    )
