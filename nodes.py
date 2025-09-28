"""
WanVideo Enhanced BlockSwap (CUDA Optimized) - Independent Node
Intelligent VRAM-DRAM balance management node definitions
"""

import torch
import logging
import sys
import os
from typing import Dict, Any, Tuple

# Handle import issues
try:
    from .intelligent_vram_manager import get_vram_manager, calculate_optimal_blockswap_config, register_model_tensors, cleanup_global_manager
except ImportError:
    # If relative import fails, try absolute import
    sys.path.insert(0, os.path.dirname(__file__))
    from intelligent_vram_manager import get_vram_manager, calculate_optimal_blockswap_config, register_model_tensors, cleanup_global_manager

# Setup logging
log = logging.getLogger(__name__)

class WanVideoEnhancedBlockSwap:
    """WanVideo Enhanced BlockSwap (CUDA Optimized) Node"""

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

        log.info(f"WanVideo Enhanced BlockSwap (CUDA Optimized)")
        log.info(f"   Auto tuning: {auto_tuning}")
        log.info(f"   VRAM threshold: {vram_threshold}%")
        log.info(f"   CUDA optimization: {enable_cuda}")

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

            # Get VRAM manager for hardware detection
            manager = get_vram_manager(vram_threshold)
            stats = manager.get_memory_stats()

            # Hardware detection and auto configuration
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                sm_count = props.multi_processor_count
                vram_total_gb = props.total_memory / (1024**3)

                log.info(f"Hardware detection:")
                log.info(f"   GPU: {props.name}")
                log.info(f"   VRAM: {vram_total_gb:.1f}GB")
                log.info(f"   SM count: {sm_count}")
                log.info(f"   Current usage: {stats.vram_usage_percent:.1f}%")

                # Auto calculate CUDA stream count
                if sm_count >= 100:  # RTX 5090/4090 high-end cards
                    auto_streams = min(16, max(8, sm_count // 10))
                elif sm_count >= 80:  # RTX 3090 etc
                    auto_streams = min(12, max(6, sm_count // 12))
                else:  # Other GPUs
                    auto_streams = min(8, max(4, sm_count // 15))

                # Auto calculate bandwidth target
                if vram_total_gb >= 30:  # Large VRAM GPU
                    auto_bandwidth = min(0.9, max(0.7, (100 - vram_threshold) / 100))
                elif vram_total_gb >= 20:  # Medium VRAM GPU
                    auto_bandwidth = min(0.8, max(0.6, (100 - vram_threshold) / 100))
                else:  # Small VRAM GPU
                    auto_bandwidth = min(0.7, max(0.5, (100 - vram_threshold) / 100))

                # Update auto-calculated parameters
                enhanced_args.update({
                    "num_cuda_streams": auto_streams,
                    "bandwidth_target": auto_bandwidth,
                })

                log.info(f"Auto configuration completed:")
                log.info(f"   CUDA streams: {auto_streams}")
                log.info(f"   Bandwidth target: {auto_bandwidth:.0%}")

            else:
                log.warning("CUDA not available, using default configuration")
        
        # Check VRAM usage
        if enable_cuda:
            try:
                manager = get_vram_manager(vram_threshold)
                current_stats = manager.get_memory_stats()

                if current_stats.vram_usage_percent > vram_threshold:
                    log.warning(f"Current VRAM usage ({current_stats.vram_usage_percent:.1f}%) exceeds threshold ({vram_threshold}%)")
                    log.warning("   Recommend enabling BlockSwap or lowering threshold")

                # Start real-time monitoring
                manager.start_monitoring()

            except Exception as e:
                log.error(f"VRAM monitoring startup failed: {e}")

        log.info(f"Enhanced BlockSwap configuration completed")
        log.info(f"   Final config: blocks={enhanced_args['blocks_to_swap']}, streams={enhanced_args['num_cuda_streams']}, ratio={enhanced_args['bandwidth_target']:.0%}")

        return (enhanced_args,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "WanVideoEnhancedBlockSwap": WanVideoEnhancedBlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoEnhancedBlockSwap": "WanVideo Enhanced BlockSwap (CUDA Optimized)",
}

# Cleanup function
def cleanup():
    """Node cleanup function"""
    cleanup_global_manager()

# Register cleanup function
import atexit
atexit.register(cleanup)
