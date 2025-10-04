"""
WanVideo Enhanced BlockSwap - Independent Node
Intelligent VRAM-DRAM balance management node definitions
Supports NVIDIA CUDA and AMD ROCm GPUs
"""

import torch
import logging
import sys
import os
from typing import Dict, Any, Tuple

# Handle import issues
try:
    from .intelligent_vram_manager import (
        get_vram_manager,
        calculate_optimal_blockswap_config,
        register_model_tensors,
        cleanup_global_manager,
        get_device_type,
        get_device_name,
        get_compute_units
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from intelligent_vram_manager import (
        get_vram_manager,
        calculate_optimal_blockswap_config,
        register_model_tensors,
        cleanup_global_manager,
        get_device_type,
        get_device_name,
        get_compute_units
    )

# Setup logging
log = logging.getLogger(__name__)

class WanVideoEnhancedBlockSwap:
    """WanVideo Enhanced BlockSwap Node (NVIDIA CUDA & AMD ROCm)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 40, "step": 1}),
                "enable_cuda_optimization": ("BOOLEAN", {"default": True}),
                "enable_dram_optimization": ("BOOLEAN", {"default": True}),
                "auto_hardware_tuning": ("BOOLEAN", {"default": True}),
                "vram_threshold_percent": ("FLOAT", {"default": 50.0, "min": 30.0, "max": 90.0, "step": 5.0}),
                "num_cuda_streams": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
                "bandwidth_target": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "offload_txt_emb": ("BOOLEAN", {"default": False}),
                "offload_img_emb": ("BOOLEAN", {"default": False}),
                "vace_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("enhanced_block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper/Enhanced"

    def setargs(self, **kwargs):
        """Set Enhanced BlockSwap parameters"""

        # Get configuration parameters
        auto_tuning = kwargs.get("auto_hardware_tuning", True)
        vram_threshold = kwargs.get("vram_threshold_percent", 50.0)
        enable_cuda = kwargs.get("enable_cuda_optimization", True)
        debug_mode = kwargs.get("debug_mode", False)

        if debug_mode:
            log.setLevel(logging.DEBUG)

        log.info(f"WanVideo Enhanced BlockSwap")
        log.info(f"   Auto tuning: {auto_tuning}")
        log.info(f"   VRAM threshold: {vram_threshold}%")
        log.info(f"   GPU optimization: {enable_cuda}")

        # Build base parameters
        enhanced_args = {
            "blocks_to_swap": kwargs.get("blocks_to_swap", 0),
            "enable_cuda_optimization": enable_cuda,
            "enable_dram_optimization": kwargs.get("enable_dram_optimization", True),
            "auto_hardware_tuning": auto_tuning,
            "vram_threshold_percent": vram_threshold,
            "num_cuda_streams": kwargs.get("num_cuda_streams", 8),
            "bandwidth_target": kwargs.get("bandwidth_target", 0.8),
            "offload_txt_emb": kwargs.get("offload_txt_emb", False),
            "offload_img_emb": kwargs.get("offload_img_emb", False),
            "vace_blocks_to_swap": kwargs.get("vace_blocks_to_swap", 0),
        }
        
        if auto_tuning:
            log.info("Enabling intelligent hardware tuning...")

            manager = get_vram_manager(vram_threshold)
            stats = manager.get_memory_stats()

            device_type = get_device_type()
            device_name = get_device_name()
            compute_units = get_compute_units()

            if device_type in ["cuda", "hip"]:
                from .intelligent_vram_manager import get_device_memory_info
                vram_total, _ = get_device_memory_info()
                vram_total_gb = vram_total / (1024**3)

                log.info(f"Hardware detection:")
                log.info(f"   Device type: {device_type.upper()}")
                log.info(f"   GPU: {device_name}")
                log.info(f"   VRAM: {vram_total_gb:.1f}GB")
                log.info(f"   Compute units: {compute_units}")
                log.info(f"   Current usage: {stats.vram_usage_percent:.1f}%")

                if compute_units >= 100:
                    auto_streams = min(16, max(8, compute_units // 10))
                elif compute_units >= 80:
                    auto_streams = min(12, max(6, compute_units // 12))
                elif compute_units >= 60:
                    auto_streams = min(10, max(6, compute_units // 10))
                else:
                    auto_streams = min(8, max(4, compute_units // 15))

                if vram_total_gb >= 30:
                    auto_bandwidth = min(0.9, max(0.7, (100 - vram_threshold) / 100))
                elif vram_total_gb >= 20:
                    auto_bandwidth = min(0.8, max(0.6, (100 - vram_threshold) / 100))
                else:
                    auto_bandwidth = min(0.7, max(0.5, (100 - vram_threshold) / 100))

                enhanced_args.update({
                    "num_cuda_streams": auto_streams,
                    "bandwidth_target": auto_bandwidth,
                })

                log.info(f"Auto configuration completed:")
                log.info(f"   Streams: {auto_streams}")
                log.info(f"   Bandwidth target: {auto_bandwidth:.0%}")

            else:
                log.warning("GPU not available, using default configuration")
        
        if enable_cuda:
            try:
                manager = get_vram_manager(vram_threshold)
                current_stats = manager.get_memory_stats()

                if current_stats.vram_usage_percent > vram_threshold:
                    log.warning(f"Current VRAM usage ({current_stats.vram_usage_percent:.1f}%) exceeds threshold ({vram_threshold}%)")
                    log.warning("   Recommend enabling BlockSwap or lowering threshold")

            except Exception as e:
                log.error(f"VRAM check failed: {e}")

        log.info(f"Enhanced BlockSwap configuration completed")
        log.info(f"   Final config: blocks={enhanced_args['blocks_to_swap']}, streams={enhanced_args['num_cuda_streams']}, ratio={enhanced_args['bandwidth_target']:.0%}")

        return (enhanced_args,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "WanVideoEnhancedBlockSwap": WanVideoEnhancedBlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoEnhancedBlockSwap": "WanVideo Enhanced BlockSwap",
}

# Cleanup function
def cleanup():
    """Node cleanup function"""
    cleanup_global_manager()

# Register cleanup function
import atexit
atexit.register(cleanup)
