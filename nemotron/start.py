import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer
import time
from cprint import c_print

from .utils import print_max_memory
from .llm import NemotronHForCausalLM

# MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"

current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, "sample_text.txt"), "r") as f:
    prompt = f.read()


def setup_hooks(model):
    def hook(w):
        # print(w.grad.norm())
        w.grad = None
        return

    for n, p in model.named_parameters():
        p.register_post_accumulate_grad_hook(hook)


def calculate_loss(model: NemotronHForCausalLM, text, tokenizer, device, max_tokens):
    tokenizer_kwargs = {"return_tensors": "pt"}
    if max_tokens is not None:
        tokenizer_kwargs.update(
            {
                "max_length": max_tokens,
                "truncation": True,
            }
        )

    inputs = tokenizer(text, **tokenizer_kwargs).to(device)
    assert inputs["input_ids"].shape[-1] == max_tokens
    # For causal LM training, labels are usually the same token IDs as inputs.
    # The model shifts them internally when computing next-token loss.
    labels = inputs["input_ids"].clone()

    outputs = model(**inputs, labels=labels, use_cache=False)
    return outputs.loss


def run_tests(model: NemotronHForCausalLM, tokenizer, device, train_tokens):
    model.train()

    model.config.sparse_ffn = False
    model.config.use_ckpt = True
    # sparse_data = TensorBuffer(60_000_000)
    sparse_data = None
    model.config.pack_sbit = False
    model.config.sparse_data = sparse_data

    c_print(f'{train_tokens=}, sparse={model.config.sparse_ffn}', color="bright_yellow")

    # Warmup
    # c_print("Starting Warmup", color="cyan")
    for _ in range(2):
        if sparse_data is not None:
            sparse_data.reset_buffer()
        loss = calculate_loss(model, prompt, tokenizer, device, max_tokens=train_tokens)
        loss.backward()
        model.zero_grad()

    # Timing
    # c_print("Starting Timing Run", color="cyan")
    torch.cuda.synchronize()
    st = time.perf_counter()
    for _ in range(10):
        if sparse_data is not None:
            sparse_data.reset_buffer()
        loss = calculate_loss(model, prompt, tokenizer, device, max_tokens=train_tokens)
        loss.backward()

    torch.cuda.synchronize()
    et = time.perf_counter()

    total_time = 1000 * (et - st) / 10
    print(f"Total Time: {total_time:.4f} ms")

    # Record memory usage
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if sparse_data is not None:
        sparse_data.reset_buffer()
    loss = calculate_loss(model, prompt, tokenizer, device, max_tokens=train_tokens)
    torch.cuda.synchronize()
    print_max_memory("After forward pass")
    loss.backward()
    torch.cuda.synchronize()
    vram = print_max_memory("After backward pass")

    # Validation
    print("-" * 50)
    print(f'Loss: {loss.detach().cpu() = }')

    if sparse_data is not None:
        if sparse_data.offset > sparse_data.size:
            c_print(
                f"Warning: Too many values detected, sparse_data.offset={sparse_data.offset.cpu().item()}, {sparse_data.size = }. "
                f"Results may be incorrect and the program may crash unexpectedly.", color="bright_red")

    return total_time, vram

def main():
    import csv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: NemotronHForCausalLM = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype, trust_remote_code=True, use_kernels=True
    ).to(device)
    setup_hooks(model)


    with open("./checkpoint.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_tokens", "vram", "time"])
        # token_sizes = [1100]
        token_sizes = [50, 100, 200, 300, 400, 500, 700, 900, 1100, 1300, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
        for train_tokens in token_sizes:
            time, vram = run_tests(model, tokenizer, device, train_tokens)
            writer.writerow([train_tokens, vram, time])
            f.flush()



    # train_tokens = 500
    # run_tests(model, tokenizer, device, train_tokens)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    main()
