from nvidia import nvcomp
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn.functional as F
import csv

from experiments.experiment import FFNReluABC, FFNRelu2ABC
from lib_sparse.fp8 import matmul, to_fp8
from lib_sparse.config import RELU2_SCALE

algos = ["LZ4", "Zstd", "Cascaded", "Bitcomp"]
ALGO = None

USE_FP8 = False

class Compressor:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def compress_tensor(self, x: torch.Tensor, scale: Tensor | None = None):
        assert x.is_cuda
        assert x.numel() > 0

        x = x.detach().contiguous()
        raw = x.view(torch.uint8).reshape(-1)

        stream = torch.cuda.current_stream(x.device).cuda_stream
        device_id = x.device.index

        codec = nvcomp.Codec(
            algorithm=self.algorithm,
            device_id=device_id,
            cuda_stream=stream,
        )

        src = nvcomp.as_array(raw, cuda_stream=stream)

        # Let PyTorch own the compressed allocation.
        max_size = codec.get_max_comp_buffer_size(src)

        storage = torch.empty(
            max_size,
            dtype=torch.uint8,
            device=x.device,
        )

        dst = nvcomp.as_array(storage, cuda_stream=stream)

        compressed_nv = codec.encode(src, out=dst)

        # Only this prefix contains valid compressed data.
        compressed = storage[:compressed_nv.buffer_size].clone()

        meta = {
            "shape": x.shape,
            "dtype": x.dtype,
            "nbytes": raw.numel(),
            "algorithm": self.algorithm,
            "scale": scale,
        }

        return compressed, meta


    def decompress_tensor(self, compressed, meta):
        assert compressed.is_cuda

        device = compressed.device
        stream = torch.cuda.current_stream(device).cuda_stream

        codec = nvcomp.Codec(algorithm=meta["algorithm"], device_id=device.index, cuda_stream=stream)

        # This should now work, because compressed is backed by
        # a normal PyTorch CUDA allocation.
        compressed_nv = nvcomp.as_array(compressed, cuda_stream=stream)

        raw = torch.empty(meta["nbytes"], dtype=torch.uint8, device=device)

        raw_nv = nvcomp.as_array(raw, cuda_stream=stream,)

        codec.decode(compressed_nv,out=raw_nv,)

        x = raw.view(meta["dtype"]).reshape(meta["shape"])

        # Dequantize fp8 back to bf16 (fp8 tensors lack mul/compare ops).
        if meta["dtype"] == torch.float8_e4m3fn:
            x = x.to(torch.bfloat16) * meta["scale"]

        return x


class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, compressor: Compressor):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        ctx.compressor = compressor
        fp8 = USE_FP8

        h = z.relu_()

        # Quantize to fp8 + scale once
        if fp8:
            h_fp8, h_scale = to_fp8(h)
        else:
            h_fp8, h_scale = h, None

        h_sparse = compressor.compress_tensor(h_fp8, h_scale)
        ctx.h_sparse = h_sparse
        y = matmul(h_fp8, W.T, fp8, a_scale=h_scale)
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        compressor: Compressor = ctx.compressor
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]
        compressed, meta = ctx.h_sparse
        ctx.h_sparse = None

        h = compressor.decompress_tensor(compressed, meta)
        del compressed, meta
        fp8 = USE_FP8

        # Use fp8 grad_output (quantize once, reuse for both matmuls).
        if fp8:
            grad_output, scale = to_fp8(grad_output)
        else:
            grad_output, scale = grad_output, None

        grad_W2 = matmul(grad_output.T, h, fp8, a_scale=scale)

        # Gradients for input
        if needs_z:
            grad_h = matmul(grad_output, W, fp8, a_scale=scale)
            grad_z = grad_h * (h > 0)
        else:
            grad_z = None
        return grad_z, grad_W2, None


class FFNRelu:
    """ FFN block with relu activation"""
    @staticmethod
    def apply(x, W1, W2, compressor: Compressor):
        """ FFN block with relu2 activation, 2 linear layers.
            x.shape = [*bs, d_in]
            W1.shape = [d_ff, d_in]
            W2.shape = [d_out, d_ff]

            out.shape = [*bs, d_out]
        """
        bs_dims = x.shape[:-1]          # [*bs, d_in]
        x = x.reshape(-1, x.shape[-1])  # [batch, d_in]

        z = matmul(x, W1.T, USE_FP8)
        y = ReluLinear.apply(z, W2, compressor)

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        return y


class Relu2Linear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, compressor: Compressor):
        """ relu(Wx) layer. """
        ctx.compressor = compressor
        ctx.save_for_backward(W)
        fp8 = USE_FP8

        r = z.relu_()

        # Cache r = relu(preact) as fp8 + scale
        if fp8:
            r_fp8, r_scale = to_fp8(r)
        else:
            r_fp8, r_scale = r, None

        h_sparse = compressor.compress_tensor(r_fp8, r_scale)
        ctx.h_sparse = h_sparse
        r.square_()
        r.mul_(RELU2_SCALE)
        y = matmul(r, W.T, fp8)
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

        r = compressor.decompress_tensor(compressed, meta)
        del compressed, meta
        fp8 = USE_FP8

        # Use fp8 grad_output (quantize once, reuse for both matmuls).
        if fp8:
            grad_output, scale = to_fp8(grad_output)
        else:
            grad_output, scale = grad_output, None

        # Reconstruct k * r² in BF16 (squaring overflows FP8), like AspRelu2B.
        grad_W2 = matmul(grad_output.T, r.square().mul_(RELU2_SCALE), fp8, a_scale=scale)

        # Needs gradient for z
        if needs_z:
            grad_h = matmul(grad_output, W, fp8, a_scale=scale)
            grad_z = 2 * RELU2_SCALE * grad_h * r
        else:
            grad_z = None

        return grad_z, grad_W2, None


class FFNRelu2:
    @staticmethod
    def apply(x, W1, W2, compressor: Compressor  ):
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

        z = matmul(x, W1.T, USE_FP8)    # [batch, d_ff]
        y = Relu2Linear.apply(z, W2, compressor) # [batch, d_out]

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        return y


class FFNReluNVCOMP(FFNReluABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, dim)
        self.compressor = Compressor(ALGO)

    def forward(self, x, pack_sbit, buffer, storage_dtype=torch.bfloat16):
        """Run the residual FFN stack while allocating sparse storage for this pass."""

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            x = x + FFNRelu.apply(x_inner, W1, W2, self.compressor)
        return x


class FFNRelu2NVCOMP(FFNRelu2ABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, dim)
        self.compressor = Compressor(ALGO)

    def forward(self, x, pack_sbit, buffer, storage_dtype=torch.bfloat16):
        """Run the residual FFN stack while allocating sparse storage for this pass."""

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            x = x + FFNRelu2.apply(x_inner, W1, W2, self.compressor)
        return x


if __name__ == "__main__":
    from experiments.experiment import evaluate_nobase

    print(f'running with relu')
    with open("./results/relu_nvcomp.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "vram", "avg_time",
        ])

        for algo in algos:
            ALGO = algo
            print(f"Running with {ALGO}")
            for _ in range(5):
                vram, time = evaluate_nobase(FFNReluNVCOMP, warmup_steps=1, eval_steps=2, bs=16000, sp_blocks=0)
                writer.writerow([algo, vram, time])
                f.flush()

    print(f'Running with relu2')
    with open("./results/relu2_nvcomp.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "vram", "avg_time",
        ])

        for algo in algos:
            ALGO = algo
            print(f"Running with {ALGO}")
            for _ in range(5):
                vram, time = evaluate_nobase(FFNRelu2NVCOMP, warmup_steps=1, eval_steps=2, bs=16000, sp_blocks=0)
                writer.writerow([algo, vram, time])
                f.flush()

            # exit(7)
