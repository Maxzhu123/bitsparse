import torch
import torch.nn.functional as F

from lib_sparse.layers import FFNRelu, FFNRelu2
from lib_sparse.bitsparse import TensorBuffer
from experiments.experiment import FFNReluABC, FFN, FFNRelu2ABC, FFNRelu2_2

BASIC_MODE = True


class FFNReluModel(FFNReluABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, layers, sp_blocks, dim)

    def forward(self, x, pack_sbit: bool, buffer: TensorBuffer|None = None,
                storage_dtype: torch.dtype = torch.bfloat16):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        if buffer is not None:
            buffer.reset_buffer()

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            if i < self.sp_blocks:
                x = x + FFNRelu.apply(x_inner, W1, W2, sparse_data=buffer, pack_sbit=pack_sbit,
                                      dtype=storage_dtype)
                # x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
            else:
                x = x + FFN.apply(x_inner, W1, W2)
        return x


class FFNRelu2Model(FFNRelu2ABC):
    def __init__(self, layers, sp_blocks, dim, dtype):
        super().__init__(dtype, layers, sp_blocks, dim)

    def forward(self, x, pack_sbit: bool, buffer: TensorBuffer|None = None,
                storage_dtype: torch.dtype = torch.bfloat16):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        if buffer is not None:
            buffer.reset_buffer()

        for i, (W1, W2) in enumerate(zip(self.W1s, self.W2s)):
            x_inner = F.rms_norm(x, x.shape[1:])
            if i < self.sp_blocks:
                x = x + FFNRelu2.apply(x_inner, W1, W2, sparse_data=buffer, pack_sbit=pack_sbit,
                                       storage_dtype=storage_dtype)
            else:
                x = x + FFNRelu2_2.apply(x_inner, W1, W2)
        return x


if __name__ == "__main__":
    from experiments.experiment import run_batch, run_layers
    import experiments.experiment as exp

    exp.DATA_SPARSITY = "Normal"
    for _ in range(5):
        run_batch(FFNReluModel, sp_blocks=10, warmup_steps=1, eval_steps=3, batch_sizes=[16_000], save_name="./results/relu_normal.csv")
    print(":"*75)
    print("Running with FFNRelu2")
    for _ in range(5):
        run_batch(FFNRelu2Model, sp_blocks=10, warmup_steps=1, eval_steps=3, batch_sizes=[16_000], save_name="./results/relu2_normal.csv")

    # exp.DATA_SPARSITY = "Sparse"
    # for _ in range(5):
    #     run_batch(FFNReluModel, sp_blocks=10, warmup_steps=1, eval_steps=3, batch_sizes=[16_000], save_name="./results/relu_sparse.csv")
    # print(":"*75)
    # print("Running with FFNRelu2")
    # for _ in range(5):
    #     run_batch(FFNRelu2Model, sp_blocks=10, warmup_steps=1, eval_steps=3, batch_sizes=[16_000], save_name="./results/relu2_sparse.csv")

    # run_layers(FFNReluModel, bs=16_000, save_name="relu2_sparser_layers.csv")
    # evaluate(FFMReluModel, bs=16_000, sp_blocks=0)
