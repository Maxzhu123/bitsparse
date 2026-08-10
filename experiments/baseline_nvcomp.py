from nvidia import nvcomp
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn.functional as F
import csv

from experiments.experiment import FFNReluABC, FFN, FFNRelu2ABC

algos = ["LZ4", "Zstd", "Cascaded", "Bitcomp"]
ALGO = None

class Compressor:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def compress_tensor(self, x: torch.Tensor):
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

        return raw.view(meta["dtype"]).reshape(meta["shape"])


class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, compressor: Compressor):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        ctx.compressor = compressor

        h = z.relu_()
        h_sparse = compressor.compress_tensor(h)
        # print(h_sparse[0].shape)
        ctx.h_sparse = h_sparse
        y = h @ W.T
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
    def apply(x, W1, W2, compressor: Compressor):
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


class Relu2Linear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, compressor: Compressor):
        """ relu(Wx) layer. """
        ctx.compressor = compressor
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = compressor.compress_tensor(h)
        ctx.h_sparse = h_sparse
        h.square_()
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
        del compressed, meta

        grad_W2 = torch.mm(grad_output.t(), h.square())

        # Needs gradient for z
        if needs_z:
            grad_h = grad_output @ W
            grad_z = 2 * grad_h * h
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

        z = x @ W1.T                    # [batch, d_ff]
        y = Relu2Linear.apply(z, W2, compressor) # [batch, d_out]

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        return y


class FFNReluNVCOMP(FFNReluABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, dim)
        self.compressor = Compressor(ALGO)

    def forward(self, x, pack_15bit, buffer):
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

    def forward(self, x, pack_15bit, buffer):
        """Run the residual FFN stack while allocating sparse storage for this pass."""

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            x = x + FFNRelu2.apply(x_inner, W1, W2, self.compressor)
        return x


if __name__ == "__main__":
    from experiment import evaluate_nobase

    with open("./results/relu2_nvcomp_16k.csv", "a", newline="") as f:
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

