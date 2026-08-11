import torch.nn.functional as F

from lib_sparse.layers import FFNRelu2
from lib_sparse.bitsparse import TensorBuffer
from experiments.experiment import FFNRelu2ABC, FFNRelu2_2, evaluate


BASIC_MODE = True

class FFNRelu2Model(FFNRelu2ABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        super().__init__(dtype, sp_blocks, layers, dim)

    def forward(self, x, pack_sbit: bool, buffer: TensorBuffer|None = None):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        if buffer is not None:
            buffer.reset_buffer()

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            if i < self.sp_blocks:
                x = x + FFNRelu2.apply(x_inner, W1, W2, sparse_data=buffer, pack_sbit=pack_sbit)
            else:
                x = x + FFNRelu2_2.apply(x_inner, W1, W2)
        return x


if __name__ == "__main__":
    from experiment import run_batch, run_layers

    # run_batch(FFNRelu2Model, save_name="relu2_sparser.csv")
    # run_layers(FFNRelu2Model, bs=16_000, save_name="relu2_sparser_layers.csv")
    evaluate(FFNRelu2Model, bs=16_000, sp_blocks=0)
