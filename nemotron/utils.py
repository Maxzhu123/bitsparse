import torch


def print_max_memory(msg):
    memory = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    print(f'{msg}: {memory:.2f} MB')

    return memory


def print_memory(msg):
    memory = torch.cuda.memory_allocated("cuda") / 1024 ** 2
    print(f'{msg}: {memory:.2f} MB')