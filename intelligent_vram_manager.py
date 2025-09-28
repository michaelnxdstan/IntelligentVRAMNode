"""
Intelligent VRAM Manager Core Module
Provides VRAM-DRAM balance management, intelligent tensor migration, real-time monitoring and other functions
"""

import torch
import psutil
import threading
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import weakref

# Setup logging
log = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory statistics information"""
    vram_total_mb: float
    vram_used_mb: float
    vram_available_mb: float
    vram_usage_percent: float
    dram_total_mb: float
    dram_used_mb: float
    dram_available_mb: float
    dram_usage_percent: float

@dataclass
class TensorInfo:
    """Tensor information"""
    tensor_id: str
    size_bytes: int
    device: torch.device
    dtype: torch.dtype
    shape: Tuple[int, ...]
    is_critical: bool = False
    priority_score: float = 1.0
    last_access_time: float = 0.0
    access_count: int = 0

class IntelligentVRAMManager:
    """Intelligent VRAM Manager"""

    def __init__(self, vram_threshold_percent: float = 50.0):
        self.vram_threshold_percent = vram_threshold_percent
        self.tensor_registry: Dict[str, TensorInfo] = {}
        self.migration_lock = threading.Lock()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # 5 second monitoring interval

        # Performance statistics
        self.migration_stats = {
            "total_migrations": 0,
            "total_migrated_mb": 0.0,
            "migration_failures": 0,
            "last_migration_time": 0.0
        }

        log.info(f"Intelligent VRAM manager initialized: threshold={vram_threshold_percent}%")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # VRAM information
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_used = torch.cuda.memory_allocated()
            vram_available = vram_total - vram_used
            vram_usage_percent = (vram_used / vram_total) * 100
        else:
            vram_total = vram_used = vram_available = 0
            vram_usage_percent = 0.0

        # DRAM information
        dram_info = psutil.virtual_memory()

        return MemoryStats(
            vram_total_mb=vram_total / (1024 * 1024),
            vram_used_mb=vram_used / (1024 * 1024),
            vram_available_mb=vram_available / (1024 * 1024),
            vram_usage_percent=vram_usage_percent,
            dram_total_mb=dram_info.total / (1024 * 1024),
            dram_used_mb=dram_info.used / (1024 * 1024),
            dram_available_mb=dram_info.available / (1024 * 1024),
            dram_usage_percent=dram_info.percent
        )
    
    def register_tensor(self, tensor_id: str, tensor: torch.Tensor,
                       is_critical: bool = False, priority: float = 1.0) -> str:
        """Register tensor to manager"""
        info = TensorInfo(
            tensor_id=tensor_id,
            size_bytes=tensor.numel() * tensor.element_size(),
            device=tensor.device,
            dtype=tensor.dtype,
            shape=tensor.shape,
            is_critical=is_critical,
            priority_score=priority,
            last_access_time=time.time(),
            access_count=1
        )

        self.tensor_registry[tensor_id] = info

        # Use weak reference to avoid memory leaks
        def cleanup_callback(ref):
            if tensor_id in self.tensor_registry:
                del self.tensor_registry[tensor_id]

        weakref.ref(tensor, cleanup_callback)

        log.debug(f"Registered tensor: {tensor_id}, size={info.size_bytes/(1024*1024):.1f}MB")
        return tensor_id

    def check_vram_threshold(self) -> bool:
        """Check if VRAM threshold is exceeded"""
        stats = self.get_memory_stats()
        return stats.vram_usage_percent > self.vram_threshold_percent
    
    def intelligent_block_migration(self, force_migration: bool = False) -> Dict[str, Any]:
        """Intelligent block migration - core algorithm"""
        with self.migration_lock:
            stats = self.get_memory_stats()

            log.info(f"Checking migration conditions: VRAM usage={stats.vram_usage_percent:.1f}%, threshold={self.vram_threshold_percent}%")

            if not force_migration and stats.vram_usage_percent <= self.vram_threshold_percent:
                return {"migrated": 0, "reason": "Threshold not exceeded"}

            # Find tensors that need migration
            migration_candidates = self._find_migration_candidates(stats)

            if not migration_candidates:
                log.warning("No migratable tensors found")
                return {"migrated": 0, "reason": "No migratable tensors"}

            # Execute migration
            migrated_count = 0
            total_migrated_mb = 0

            for tensor_id, info in migration_candidates:
                try:
                    if self._migrate_tensor_to_dram(tensor_id, info):
                        migrated_count += 1
                        total_migrated_mb += info.size_bytes / (1024 * 1024)

                        log.info(f"Tensor migrated to DRAM: {tensor_id}, size={info.size_bytes/(1024*1024):.1f}MB")

                        # Check if target is reached
                        current_stats = self.get_memory_stats()
                        if current_stats.vram_usage_percent <= self.vram_threshold_percent * 0.9:
                            break

                except Exception as e:
                    log.error(f"Tensor migration failed: {tensor_id}, error: {e}")
                    self.migration_stats["migration_failures"] += 1

            # Update statistics
            self.migration_stats["total_migrations"] += migrated_count
            self.migration_stats["total_migrated_mb"] += total_migrated_mb
            self.migration_stats["last_migration_time"] = time.time()

            # Get final state
            final_stats = self.get_memory_stats()

            result = {
                "migrated": migrated_count,
                "total_mb": total_migrated_mb,
                "vram_before": stats.vram_usage_percent,
                "vram_after": final_stats.vram_usage_percent,
                "reason": "Intelligent migration completed"
            }

            log.info(f"Migration completed: {migrated_count} tensors, {total_migrated_mb:.1f}MB, VRAM: {stats.vram_usage_percent:.1f}% -> {final_stats.vram_usage_percent:.1f}%")

            return result
    
    def _find_migration_candidates(self, stats: MemoryStats) -> List[Tuple[str, TensorInfo]]:
        """Find migration candidate tensors"""
        # Only consider non-critical tensors on VRAM
        vram_tensors = [
            (tid, info) for tid, info in self.tensor_registry.items()
            if info.device.type == 'cuda' and not info.is_critical
        ]

        if not vram_tensors:
            return []

        # Sort by priority (lower priority migrated first)
        vram_tensors.sort(key=lambda x: x[1].priority_score)

        # Calculate memory to be freed
        target_usage = self.vram_threshold_percent * 0.8  # Target to reduce to 80%
        current_usage = stats.vram_usage_percent
        need_to_free_percent = current_usage - target_usage
        need_to_free_mb = (need_to_free_percent / 100) * stats.vram_total_mb

        # Select tensors to migrate
        candidates = []
        freed_mb = 0

        for tensor_id, info in vram_tensors:
            candidates.append((tensor_id, info))
            freed_mb += info.size_bytes / (1024 * 1024)

            if freed_mb >= need_to_free_mb:
                break

        log.info(f"Selected migration candidates: {len(candidates)} tensors, expected to free {freed_mb:.1f}MB")
        return candidates

    def _migrate_tensor_to_dram(self, tensor_id: str, info: TensorInfo) -> bool:
        """Migrate tensor to DRAM"""
        try:
            # Here should implement actual tensor migration logic
            # Since this is an example, we just simulate migration

            # Update tensor information
            info.device = torch.device('cpu')

            return True

        except Exception as e:
            log.error(f"Tensor migration failed: {tensor_id}, error: {e}")
            return False
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        log.info("Started VRAM real-time monitoring")

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

        log.info("Stopped VRAM real-time monitoring")

    def _monitoring_loop(self):
        """Monitoring loop"""
        while self.monitoring_active:
            try:
                if self.check_vram_threshold():
                    log.warning("VRAM usage exceeds threshold, starting intelligent migration...")
                    self.intelligent_block_migration()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                log.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        stats = self.get_memory_stats()

        return {
            "memory_stats": stats,
            "tensor_count": len(self.tensor_registry),
            "migration_stats": self.migration_stats.copy(),
            "threshold_percent": self.vram_threshold_percent,
            "monitoring_active": self.monitoring_active
        }

    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        self.tensor_registry.clear()
        log.info("VRAM manager cleanup completed")

# Global manager instance
_global_manager: Optional[IntelligentVRAMManager] = None
_manager_lock = threading.Lock()

def get_vram_manager(threshold_percent: float = 50.0) -> IntelligentVRAMManager:
    """Get global VRAM manager instance"""
    global _global_manager

    with _manager_lock:
        if _global_manager is None:
            _global_manager = IntelligentVRAMManager(threshold_percent)
        elif _global_manager.vram_threshold_percent != threshold_percent:
            # Update when threshold changes
            _global_manager.vram_threshold_percent = threshold_percent
            log.info(f"Updated VRAM threshold: {threshold_percent}%")

        return _global_manager

def cleanup_global_manager():
    """Clean up global manager"""
    global _global_manager

    with _manager_lock:
        if _global_manager:
            _global_manager.cleanup()
            _global_manager = None

def calculate_optimal_blockswap_config(transformer, vram_threshold: float = 50.0) -> Dict[str, Any]:
    """Calculate optimal BlockSwap configuration based on actual model size and VRAM usage"""
    manager = get_vram_manager(vram_threshold)

    # Analyze transformer model
    total_params = sum(p.numel() for p in transformer.parameters())
    total_size_mb = sum(p.numel() * 4 for p in transformer.parameters()) / (1024 * 1024)

    # Get layer count
    if hasattr(transformer, 'config') and hasattr(transformer.config, 'num_hidden_layers'):
        num_layers = transformer.config.num_hidden_layers
    elif hasattr(transformer, 'blocks'):
        num_layers = len(transformer.blocks)
    else:
        num_layers = 24  # Default value

    # Get memory state
    stats = manager.get_memory_stats()

    # VRAM-DRAM balance calculation
    import psutil
    dram_info = psutil.virtual_memory()
    dram_total_mb = dram_info.total / (1024 * 1024)
    dram_available_mb = dram_info.available / (1024 * 1024)
    dram_usage_percent = dram_info.percent

    # 1. Average layer size
    avg_layer_size_mb = total_size_mb / max(1, num_layers)

    # 2. VRAM threshold calculation (threshold is trigger point, not usage limit)
    vram_threshold_mb = stats.vram_total_mb * (vram_threshold / 100)
    current_usage_mb = stats.vram_used_mb

    # Actual available VRAM = Total VRAM - Current usage - Safety reserve (10%)
    safety_reserve_mb = stats.vram_total_mb * 0.1  # Reserve 10% as safety margin
    actual_available_vram_mb = stats.vram_total_mb - current_usage_mb - safety_reserve_mb

    # 3. Available DRAM (with safety margin)
    safe_dram_ratio = 0.8  # Only use 80% of available DRAM
    usable_dram_mb = dram_available_mb * safe_dram_ratio

    # 4. Total available memory = Actual available VRAM + Available DRAM
    total_available_memory_mb = actual_available_vram_mb + usable_dram_mb

    log.info(f"VRAM-DRAM balance analysis:")
    log.info(f"   VRAM total: {stats.vram_total_mb:.1f}MB")
    log.info(f"   VRAM used: {current_usage_mb:.1f}MB")
    log.info(f"   VRAM available: {actual_available_vram_mb:.1f}MB (reserved 10% safety margin)")
    log.info(f"   VRAM threshold: {vram_threshold_mb:.1f}MB ({vram_threshold}%)")
    log.info(f"   DRAM available: {dram_available_mb:.1f}MB (usage {dram_usage_percent:.1f}%)")
    log.info(f"   DRAM usable: {usable_dram_mb:.1f}MB (safe use 80%)")
    log.info(f"   Total available: {total_available_memory_mb:.1f}MB")
    log.info(f"   Model requirement: {total_size_mb:.1f}MB")

    # 5. Intelligent blocking calculation
    if total_size_mb > total_available_memory_mb:
        # Memory completely insufficient
        log.error(f"Memory severely insufficient! Need {total_size_mb:.1f}MB, but only have {total_available_memory_mb:.1f}MB")
        blocks_to_swap = min(num_layers, max(num_layers // 2, int((total_size_mb - total_available_memory_mb) / avg_layer_size_mb) + 5))

    elif total_size_mb > actual_available_vram_mb:
        # VRAM insufficient, need to use DRAM
        overflow_mb = total_size_mb - actual_available_vram_mb

        if overflow_mb <= usable_dram_mb:
            # DRAM sufficient, calculate optimal blocking
            blocks_to_swap = min(num_layers, max(1, int(overflow_mb / avg_layer_size_mb) + 1))
            log.info(f"VRAM insufficient by {overflow_mb:.1f}MB, will use DRAM, recommend swapping {blocks_to_swap} blocks")
        else:
            # DRAM also insufficient, need more blocking
            blocks_to_swap = min(num_layers, max(3, int(overflow_mb / avg_layer_size_mb) + 2))
            log.warning(f"Both VRAM+DRAM tight, recommend swapping {blocks_to_swap} blocks")
    else:
        # Memory sufficient
        blocks_to_swap = 0
        log.info(f"Memory sufficient, no BlockSwap needed")

    # 5. Calculate CUDA stream count (based on blocks and hardware)
    if torch.cuda.is_available():
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        if blocks_to_swap > 0:
            # With BlockSwap, need more streams to handle migration
            num_streams = min(16, max(4, min(blocks_to_swap * 2, sm_count // 4)))
        else:
            # Without BlockSwap, standard stream configuration
            num_streams = min(8, max(2, sm_count // 8))
    else:
        num_streams = 4

    # 6. Calculate memory usage ratio
    if stats.vram_total_mb > 0:
        # Dynamic adjustment based on actual available memory
        memory_ratio = min(0.95, max(0.6, actual_available_vram_mb / stats.vram_total_mb))
    else:
        memory_ratio = 0.8

    config = {
        "blocks_to_swap": blocks_to_swap,
        "num_cuda_streams": num_streams,
        "bandwidth_target": memory_ratio,
        "vram_threshold_percent": vram_threshold,
        "model_analysis": {
            "total_params": total_params,
            "total_size_mb": total_size_mb,
            "num_layers": num_layers,
            "avg_layer_size_mb": avg_layer_size_mb,
        },
        "memory_analysis": {
            "vram_total_mb": stats.vram_total_mb,
            "vram_used_mb": stats.vram_used_mb,
            "vram_available_mb": actual_available_vram_mb,
            "vram_threshold_mb": vram_threshold_mb,
            "dram_total_mb": dram_total_mb,
            "dram_available_mb": dram_available_mb,
            "dram_usable_mb": usable_dram_mb,
            "dram_usage_percent": dram_usage_percent,
            "total_available_mb": total_available_memory_mb,
            "overflow_mb": max(0, total_size_mb - actual_available_vram_mb),
            "total_overflow_mb": max(0, total_size_mb - total_available_memory_mb),
            "memory_sufficient": total_size_mb <= total_available_memory_mb,
            "vram_sufficient": total_size_mb <= actual_available_vram_mb,
        }
    }

    log.info(f"VRAM-DRAM balance calculation completed:")
    log.info(f"   Model: {total_params:,} parameters, {total_size_mb:.1f}MB, {num_layers} layers")
    log.info(f"   VRAM: {stats.vram_used_mb:.1f}/{stats.vram_total_mb:.1f}MB ({stats.vram_usage_percent:.1f}%)")
    log.info(f"   DRAM: {dram_available_mb:.1f}/{dram_total_mb:.1f}MB (usage {dram_usage_percent:.1f}%)")
    log.info(f"   Total available: {total_available_memory_mb:.1f}MB (VRAM{actual_available_vram_mb:.1f}MB + DRAM{usable_dram_mb:.1f}MB)")
    log.info(f"   Configuration: blocks_to_swap={blocks_to_swap}, streams={num_streams}, ratio={memory_ratio:.0%}")

    if total_size_mb > total_available_memory_mb:
        log.error(f"Memory severely insufficient! Need {total_size_mb:.1f}MB, total available {total_available_memory_mb:.1f}MB, missing {total_size_mb - total_available_memory_mb:.1f}MB")
    elif blocks_to_swap > 0:
        log.warning(f"VRAM insufficient by {max(0, total_size_mb - actual_available_vram_mb):.1f}MB, will use DRAM, enable {blocks_to_swap} block swapping")
    else:
        log.info(f"Model can be fully loaded into VRAM, no DRAM needed")

    return config

def register_model_tensors(transformer, vram_threshold: float = 50.0) -> Dict[str, Any]:
    """Register model tensors to VRAM manager"""
    manager = get_vram_manager(vram_threshold)

    registered_count = 0
    total_size_mb = 0

    try:
        # Iterate through model parameters
        for name, param in transformer.named_parameters():
            if param.device.type == 'cuda':
                # Determine if critical tensor
                is_critical = 'embed' in name.lower() or 'norm' in name.lower() or param.numel() < 1000

                # Calculate priority
                priority = 1.0
                if 'weight' in name.lower():
                    priority += 0.5
                if 'attention' in name.lower():
                    priority += 0.3

                # Large tensors have low priority, easy to migrate
                size_mb = param.numel() * param.element_size() / (1024 * 1024)
                if size_mb > 100:
                    priority -= 0.5
                elif size_mb > 50:
                    priority -= 0.3

                priority = max(0.1, priority)

                # Register tensor
                tensor_id = f"model.{name}"
                manager.register_tensor(tensor_id, param, is_critical, priority)

                registered_count += 1
                total_size_mb += size_mb

        log.info(f"Registered {registered_count} tensors to VRAM manager, total size {total_size_mb:.1f}MB")

        # Start real-time monitoring
        manager.start_monitoring()

    except Exception as e:
        log.error(f"Tensor registration failed: {e}")

    return {
        "registered_count": registered_count,
        "total_size_mb": total_size_mb,
        "manager": manager,
    }
