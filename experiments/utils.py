import torch
from cprint import c_print
from torch import nn as nn


def print_memory(msg):
    memory = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    c_print(f'{msg}: {memory:.2f} MB', color="bright_cyan")


def setup_hooks(model: nn.Module):
    """ Simulate hook optimiser that applies update + clears grads immediately."""
    def hook(w):
        w.grad = None
        return

    model.handles = []
    for n, p in model.named_parameters():
        handle = p.register_post_accumulate_grad_hook(hook)
        model.handles.append(handle)


def remove_hooks(model):
    for handle in model.handles:
        handle.remove()
