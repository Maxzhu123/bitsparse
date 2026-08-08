import torch.nn.functional as F

from lib_sparse.layers import FFNRelu, FFNRelu_3
from lib_sparse.bitsparse import TensorBuffer
from experiment import DeepFFN_abc, FFN

FFN_BLOCK_LAYERS = 2
LAYERS = 8
BATCH_SIZE = 10000
DIM = 4096

CKPT = True
BASIC_MODE = True


class FFMReluModel(DeepFFN_abc):
    handles: list

    def __init__(self, layers, sp_blocks, dtype, ckpt=CKPT):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, DIM, FFN_BLOCK_LAYERS, ckpt)

    # @torch.compile
    def forward(self, x, pack_15bit: bool, buffer: TensorBuffer|None = None):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        if buffer is not None:
            buffer.reset_buffer()
        if self.block_layers == 2:
            for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
                x_inner = F.rms_norm(x, x.shape[1:])
                if i < self.sp_blocks:
                    x = x + FFNRelu.apply(x_inner, W1, W2, sparse_data=buffer, pack_15bit=pack_15bit)
                    # x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
                else:
                    x = x + FFN.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNRelu_3.apply(x_inner, W1, W2, W3, sparse_data=buffer, pack_15bit=pack_15bit)
        return x



if __name__ == "__main__":
    from experiment import run_batch, run_layers

    run_batch(FFMReluModel, save_name="relu2_sparser.csv")
    run_layers(FFMReluModel, bs=16_000, save_name="relu2_sparser_layers.csv")
    # evaluate(FFMReluModel, bs=16_000, sp_blocks=0)