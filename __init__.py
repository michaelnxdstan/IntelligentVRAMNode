"""
WanVideo Enhanced BlockSwap (CUDA Optimized) - Independent Node
Intelligent VRAM-DRAM balance management, prevents VRAM overflow, optimizes large model inference performance
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version information
__version__ = "2.0.0"
__author__ = "eddy"
__description__ = "WanVideo Enhanced BlockSwap with Intelligent VRAM Management"

# Node information
WEB_DIRECTORY = "./web"
