"""Quantization utilities for weights and exported formats."""

from typing import Dict
import torch


def quantize_fp16(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert floating point tensors to float16."""
    fp16_dict = {}
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            fp16_dict[k] = v.half()
        else:
            fp16_dict[k] = v
    return fp16_dict


def calculate_quantization_stats(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculate parameter count and memory footprint in MB."""
    total_elements = 0
    total_bytes = 0
    for v in state_dict.values():
        total_elements += v.numel()
        total_bytes += v.numel() * v.element_size()
    return {
        "total_parameters": float(total_elements),
        "total_megabytes": float(total_bytes) / (1024 * 1024),
    }
