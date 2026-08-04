import torch
from torch import Tensor
import time

from code import compress_fn, uncompress_fn

# Generate data
def generate_data(n: int, G, dtype, device):
    x = torch.randn(n, generator=G, dtype=dtype, device=device)
    return x.abs()

def run_batch(data: Tensor, dtype, device):
    compressed = compress_fn(data, dtype, device)

    uncompressed = uncompress_fn(compressed, data.shape, dtype, device)

    return uncompressed


def main():
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = torch.Generator(device=device).manual_seed(0)

    sizes = [3**i for i in range(12, 17)] # From 16K to 49M
    print(sizes)
    iters = 100

    tot_times = 0
    for n in sizes:
        data = generate_data(n, G, dtype, device)

        # Warmup
        for i in range(5):
            run_batch(data, dtype, device)

        torch.cuda.synchronize()
        st = time.perf_counter()
        for i in range(iters):
            uncompressed = run_batch(data, dtype, device)
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Check correctness
        assert torch.all(torch.eq(uncompressed, data)), "Data correctness check failed."

        avg_time = 1000*(end - st)/iters
        tot_times += avg_time
        print(f'{n=}, Time: {avg_time:.3g}ms')

    print("Passed")
    print(f'Total time: {tot_times:.3g}ms')


if __name__ == '__main__':
    main()