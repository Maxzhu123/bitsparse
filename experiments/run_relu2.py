import torch.nn.functional as F

from lib_sparse.layers import FFNRelu2, FFNRelu2_3, FFNSparseRelu2
from lib_sparse.bitsparse import TensorBuffer
from experiments.experiment import FFN_relu2_abc, FFNRelu2_2, evaluate

LAYERS = 8
BATCH_SIZE = 10000
DIM = 4096

BASIC_MODE = True

class FFNRelu2Model(FFN_relu2_abc):
    def __init__(self, layers, sp_blocks, dtype):
        super().__init__(dtype, sp_blocks, layers, DIM, 2)

    def forward(self, x, pack_15bit: bool, buffer: TensorBuffer|None = None):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        if buffer is not None:
            buffer.reset_buffer()
        if self.block_layers == 2:
            for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
                x_inner = F.rms_norm(x, x.shape[1:])
                if i < self.sp_blocks:
                    x = x + FFNRelu2.apply(x_inner, W1, W2, sparse_data=buffer, pack_15bit=pack_15bit)
                    # x = x + FFNSparse3.apply(x_inner, W1, W2, buffer)
                else:
                    x = x + FFNRelu2_2.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu2_3.apply(x_inner, W1, W2, W3, sparse_data=buffer, pack_15bit=pack_15bit)
        return x



if __name__ == "__main__":
    from experiment import run_batch, run_layers

    run_batch(FFNRelu2Model, save_name="relu2_sparser.csv")
    run_layers(FFNRelu2Model, bs=16_000, save_name="relu2_sparser_layers.csv")
    evaluate(FFNRelu2Model, bs=16_000, sp_blocks=0)
