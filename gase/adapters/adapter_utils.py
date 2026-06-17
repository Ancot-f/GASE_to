"""Adapter utility functions: freeze/unfreeze, parameter counting, state copy."""

import copy
from typing import Iterator

import torch
from torch import nn


def freeze_module(module: nn.Module) -> None:
    """
    Freeze all parameters in a module (set requires_grad=False).

    Args:
        module: PyTorch module to freeze.
    """
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """
    Unfreeze all parameters in a module (set requires_grad=True).

    Args:
        module: PyTorch module to unfreeze.
    """
    for p in module.parameters():
        p.requires_grad = True


def count_trainable_parameters(module: nn.Module) -> int:
    """
    Count the number of trainable parameters in a module.

    Args:
        module: PyTorch module.

    Returns:
        Number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def copy_adapter_state(src: nn.Module, dst: nn.Module) -> None:
    """
    Copy adapter state from source to destination (deep copy).

    Args:
        src: source adapter module.
        dst: destination adapter module.
    """
    dst.load_state_dict(copy.deepcopy(src.state_dict()))


def reset_adapter_parameters(adapter: nn.Module) -> None:
    """
    Reset adapter parameters to initial values.

    Args:
        adapter: adapter module with a reset_parameters method.
    """
    if hasattr(adapter, "reset_parameters"):
        adapter.reset_parameters()
