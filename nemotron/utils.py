import torch
from cprint import c_print


def print_max_memory(msg):
    memory = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    c_print(f'{msg}: {memory:.2f} MB', color="bright_cyan")


def print_memory(msg):
    memory = torch.cuda.memory_allocated("cuda") / 1024 ** 2
    c_print(f'{msg}: {memory:.2f} MB', color="bright_green")