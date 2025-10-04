"""
WanVideo Enhanced BlockSwap - Independent Node
Intelligent VRAM-DRAM balance management, prevents VRAM overflow, optimizes large model inference performance
Supports both NVIDIA CUDA and AMD ROCm GPUs
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__version__ = "2.1.0"
__author__ = "eddy"
__description__ = "WanVideo Enhanced BlockSwap with Intelligent VRAM Management (NVIDIA & AMD)"

# Node information
WEB_DIRECTORY = "./web"
