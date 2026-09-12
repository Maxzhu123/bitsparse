import torch

from lib_sparse.layers import RMSFFNRelu, RMSFFNRelu2
from lib_sparse.bitsparse import TensorBuffer
from experiments.experiment import FFNReluABC, RMSFFN, FFNRelu2ABC, RMSFFNRelu2 as DenseRMSFFNRelu2

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
            if i < self.sp_blocks:
                x = x + RMSFFNRelu.apply(
                    x, W1, W2, eps=torch.finfo(torch.float32).eps,
                    sparse_data=buffer, pack_sbit=pack_sbit, storage_dtype=storage_dtype,
                )
            else:
                x = x + RMSFFN.apply(x, W1, W2)
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
            if i < self.sp_blocks:
                # Normalize raw x once, using F.rms_norm's default BF16 accumulator epsilon.
                x = x + RMSFFNRelu2.apply(
                    x, W1, W2, eps=torch.finfo(torch.float32).eps,
                    sparse_data=buffer, pack_sbit=pack_sbit, storage_dtype=storage_dtype,
                )
            else:
                x = x + DenseRMSFFNRelu2.apply(x, W1, W2)
        return x


if __name__ == "__main__":
    from cprint import c_print
    from experiments.experiment import run_batch, run_layers
    import experiments.experiment as exp

    # exp.DATA_SPARSITY =  "Normal" #"Sparse"
    # c_print(f"Data sparsity overwritten to {exp.DATA_SPARSITY}", color="bright_green")
    # print("Running with Relu")
    # run_batch(FFNReluModel, sp_blocks=99, warmup_steps=1, eval_steps=3, batch_sizes=[32, 128, 512, 2000, 4000, 8000, 16000, 32000], save_name="./results/relu_normal.csv")
    # print(":"*75)
    # print("Running with Relu2")
    # run_batch(FFNRelu2Model, sp_blocks=99, warmup_steps=1, eval_steps=3, batch_sizes=[32, 128, 512, 2000, 4000, 8000, 16000, 32000], save_name="./results/relu2_normal.csv")
    #
    # exp.DATA_SPARSITY =  "Sparse"
    # c_print(f"Data sparsity overwritten to {exp.DATA_SPARSITY}", color="bright_green")
    # print("Running with Relu")
    # run_batch(FFNReluModel, sp_blocks=99, warmup_steps=1, eval_steps=3, batch_sizes=[32, 128, 512, 2000, 4000, 8000, 16000, 32000], save_name="./results/relu_sparse.csv")
    # print(":"*75)
    # print("Running with Relu2")
    # run_batch(FFNRelu2Model, sp_blocks=99, warmup_steps=1, eval_steps=3, batch_sizes=[32, 128, 512, 2000, 4000, 8000, 16000, 32000], save_name="./results/relu2_sparse.csv")
    #
    # # Vary number of compressed layers
    # exp.DATA_SPARSITY =  "Normal" #"Sparse"
    # c_print(f"Data sparsity overwritten to {exp.DATA_SPARSITY}", color="bright_green")
    # # Evaluate relu blocks
    # print(":"*75)
    # print("Running with Relu")
    # for sp in range(exp.LAYERS+1):
    #     print(f'{sp = }')
    #     run_batch(FFNReluModel, sp_blocks=sp, warmup_steps=1, eval_steps=3, batch_sizes=[16000], save_name="./results/relu_normal_layers.csv")
    # print(":" * 75)
    # print("Running with Relu")
    # for sp in range(exp.LAYERS+1):
    #     print(f'{sp = }')
    #     run_batch(FFNRelu2Model, sp_blocks=sp, warmup_steps=1, eval_steps=3, batch_sizes=[16000], save_name="./results/relu2_normal_layers.csv")
    #
    # exp.DATA_SPARSITY =  "Sparse"
    # c_print(f"Data sparsity overwritten to {exp.DATA_SPARSITY}", color="bright_green")
    # # Evaluate relu blocks
    # print(":"*75)
    # print("Running with Relu")
    # for sp in range(exp.LAYERS+1):
    #     print(f'{sp = }')
    #     run_batch(FFNReluModel, sp_blocks=sp, warmup_steps=1, eval_steps=3, batch_sizes=[16000], save_name="./results/relu_sparse_layers.csv")
    # print(":" * 75)
    # print("Running with Relu2")
    # for sp in range(exp.LAYERS+1):
    #     print(f'{sp = }')
    #     run_batch(FFNRelu2Model, sp_blocks=sp, warmup_steps=1, eval_steps=3, batch_sizes=[16000], save_name="./results/relu2_sparse_layers.csv")


    exp.DATA_SPARSITY =  "Normal"
    c_print(f"Data sparsity overwritten to {exp.DATA_SPARSITY}", color="bright_green")
    run_layers(FFNReluModel, bs=16_000, save_name="relu_normal_fp8.csv")
    run_layers(FFNRelu2Model, bs=16_000, save_name="relu2_normal_fp8.csv")

    exp.DATA_SPARSITY =  "Sparse" #"Sparse"
    c_print(f"Data sparsity overwritten to {exp.DATA_SPARSITY}", color="bright_green")
    run_layers(FFNReluModel, bs=16_000, save_name="relu_sparse_fp8.csv")
    run_layers(FFNRelu2Model, bs=16_000, save_name="relu2_sparse_fp8.csv")

