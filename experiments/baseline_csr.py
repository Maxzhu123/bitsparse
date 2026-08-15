import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn.functional as F
import csv

from experiments.experiment import FFNReluABC, FFN, FFNRelu2ABC


def to_sparse_csr(h):
    h_sparse = h.to_sparse_csr()

    # Convert to int32 indexing
    crow, col, vals = h_sparse.crow_indices(), h_sparse.col_indices(), h_sparse.values()
    crow, col = crow.to(torch.int32), col.to(torch.int32)
    h_sparse = torch.sparse_csr_tensor(crow, col, vals)

    return h_sparse


class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)

        h = z.relu_()
        y = h @ W.T

        ctx.h_sparse = to_sparse_csr(h)
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]
        h_sparse = ctx.h_sparse
        ctx.h_sparse = None

        h = h_sparse.to_dense()
        del h_sparse
        grad_W2 = grad_output.t() @ h

        # Gradients for input
        grad_h = grad_output @ W
        grad_z = grad_h * (h > 0)
        return grad_z, grad_W2, None


class FFNRelu:
    """ FFN block with relu activation"""
    @staticmethod
    def apply(x, W1, W2):
        """ FFN block with relu2 activation, 2 linear layers.
            x.shape = [*bs, d_in]
            W1.shape = [d_ff, d_in]
            W2.shape = [d_out, d_ff]

            out.shape = [*bs, d_out]
        """
        bs_dims = x.shape[:-1]          # [*bs, d_in]
        x = x.reshape(-1, x.shape[-1])  # [batch, d_in]

        z = x @ W1.T
        y = ReluLinear.apply(z, W2)

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        return y


class Relu2Linear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = to_sparse_csr(h)
        ctx.h_sparse = h_sparse
        h.square_()
        y = h @ W.T
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        needs_z = ctx.needs_input_grad[0]

        h = ctx.h_sparse.to_dense()
        ctx.h_sparse = None
        del ctx.h_sparse

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
    def apply(x, W1, W2):
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
        y = Relu2Linear.apply(z, W2) # [batch, d_out]

        y = y.reshape(*bs_dims,  y.shape[-1])   # [*bs, d_out]
        return y


class FFNReluCSR(FFNReluABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, dim)

    @torch.compile()
    def forward(self, x, _, __, ___):
        """Run the residual FFN stack while allocating sparse storage for this pass."""

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            x = x + FFNRelu.apply(x_inner, W1, W2)
        return x


class FFNRelu2CSR(FFNRelu2ABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, dim)

    def forward(self, x, _, __, ___):
        """Run the residual FFN stack while allocating sparse storage for this pass."""

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            x = x + FFNRelu2.apply(x_inner, W1, W2)
        return x


if __name__ == "__main__":
    from experiments.experiment import evaluate_nobase

    with open("./results/relu2_csr.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "vram", "avg_time",
        ])

        for _ in range(5):
            vram, time = evaluate_nobase(FFNRelu2CSR, warmup_steps=1, eval_steps=3, bs=16000, sp_blocks=0)
            writer.writerow(["csr", vram, time])
            f.flush()

            # exit(7)
    with open("./results/relu_csr.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "vram", "avg_time",
        ])

        for _ in range(5):
            vram, time = evaluate_nobase(FFNReluCSR, warmup_steps=1, eval_steps=3, bs=16000, sp_blocks=0)
            writer.writerow(["csr", vram, time])
            f.flush()

            # exit(7)
