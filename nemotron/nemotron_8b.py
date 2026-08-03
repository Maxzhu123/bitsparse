import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llm import NemotronHForCausalLM

MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    ).to(device)
    model.eval()

    prompt = "When was NVIDIA founded?"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            # Transformers 5.12 initializes a generic cache before Nemotron's
            # remote code can create its required hybrid Mamba/attention cache.
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
