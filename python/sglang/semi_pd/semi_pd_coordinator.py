# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Semi-PD coordinator that integrates all components."""

import logging
import threading
import time
from typing import Dict, Optional

import torch

from sglang.semi_pd.process_rotation_manager import ProcessRotationManager, SMAllocation
from sglang.semi_pd.slo_aware_resource_manager import (
    SLOAwareResourceManager,
    SLOTarget,
    WorkloadMetrics,
)
from sglang.semi_pd.unified_storage_manager import UnifiedStorageManager
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


class SemiPDCoordinator:
    """
    Semi-PD协调器
    
    整合并协调以下组件：
    1. ProcessRotationManager: 进程轮换管理
    2. UnifiedStorageManager: 统一存储管理
    3. SLOAwareResourceManager: SLO感知资源管理
    
    提供统一的管理接口和监控能力。
    """
    
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        initial_sm_allocation: SMAllocation,
        slo_target: SLOTarget,
        total_kv_blocks: int,
        kv_block_size: int,
        device: torch.device,
    ):
        self.server_args = server_args
        self.port_args = port_args
        self.device = device
        
        # 初始化组件
        logger.info("Initializing Semi-PD coordinator components")
        
        # 1. 进程轮换管理器
        self.process_rotation_manager = ProcessRotationManager(
            server_args=server_args,
            port_args=port_args,
            initial_sm_allocation=initial_sm_allocation,
        )
        
        # 2. 统一存储管理器
        self.storage_manager = UnifiedStorageManager(
            total_blocks=total_kv_blocks,
            block_size=kv_block_size,
            device=device,
            enable_prefix_caching=server_args.enable_prefix_caching,
        )
        
        # 3. SLO感知资源管理器
        self.slo_manager = SLOAwareResourceManager(
            slo_target=slo_target,
            process_rotation_manager=self.process_rotation_manager,
            window_size_seconds=30.0,
            adjustment_cooldown_seconds=60.0,
        )
        
        # 状态管理
        self.running = False
        self.start_time = None
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_allocations": 0,
            "failed_allocations": 0,
            "resource_adjustments": 0,
            "slo_violations": 0,
        }
        
        logger.info("Semi-PD coordinator initialized successfully")
        
    def start(self):
        """启动协调器"""
        logger.info("Starting Semi-PD coordinator")
        
        try:
            self.running = True
            self.start_time = time.time()
            
            # 启动各个组件
            self.process_rotation_manager.start()
            self.slo_manager.start()
            
            logger.info("Semi-PD coordinator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Semi-PD coordinator: {e}")
            self.stop()
            raise
            
    def stop(self):
        """停止协调器"""
        logger.info("Stopping Semi-PD coordinator")
        
        self.running = False
        
        try:
            # 停止各个组件
            if hasattr(self, 'slo_manager'):
                self.slo_manager.stop()
                
            if hasattr(self, 'process_rotation_manager'):
                self.process_rotation_manager.stop()
                
            logger.info("Semi-PD coordinator stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Semi-PD coordinator: {e}")
            
    def allocate_kv_blocks(
        self, 
        request_id: str, 
        num_blocks: int,
        requester_role: InstanceRole = InstanceRole.OTHER,
    ) -> Optional[list]:
        """
        分配KV cache块
        
        Args:
            request_id: 请求ID
            num_blocks: 需要的块数量
            requester_role: 请求者角色
            
        Returns:
            分配的块句柄列表，失败时返回None
        """
        try:
            self.stats["total_requests"] += 1
            
            # 通过存储管理器分配块
            block_ids = self.storage_manager.allocate_blocks(
                request_id=request_id,
                num_blocks=num_blocks,
                requester_role=requester_role,
            )
            
            if block_ids:
                self.stats["successful_allocations"] += 1
                logger.debug(f"Allocated {num_blocks} blocks for request {request_id}")
                
                # 获取块句柄
                handles = self.storage_manager.get_block_handles(request_id)
                return handles
            else:
                self.stats["failed_allocations"] += 1
                logger.warning(f"Failed to allocate {num_blocks} blocks for request {request_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error allocating blocks for request {request_id}: {e}")
            self.stats["failed_allocations"] += 1
            return None
            
    def deallocate_kv_blocks(self, request_id: str) -> bool:
        """
        释放KV cache块
        
        Args:
            request_id: 请求ID
            
        Returns:
            是否成功释放
        """
        try:
            success = self.storage_manager.deallocate_blocks(request_id)
            
            if success:
                logger.debug(f"Deallocated blocks for request {request_id}")
            else:
                logger.warning(f"Failed to deallocate blocks for request {request_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error deallocating blocks for request {request_id}: {e}")
            return False
            
    def report_workload_metrics(self, metrics: WorkloadMetrics):
        """
        报告工作负载指标
        
        Args:
            metrics: 工作负载指标
        """
        try:
            # 传递给SLO管理器
            self.slo_manager.report_metrics(metrics)
            
            # 检查SLO违反
            if (metrics.ttft_p95 > self.slo_manager.slo_target.ttft_target_ms or
                metrics.tpot_p95 > self.slo_manager.slo_target.tpot_target_ms):
                self.stats["slo_violations"] += 1
                
        except Exception as e:
            logger.error(f"Error reporting workload metrics: {e}")
            
    def request_resource_adjustment(self, new_allocation: SMAllocation) -> bool:
        """
        请求资源调整
        
        Args:
            new_allocation: 新的SM分配
            
        Returns:
            是否成功提交请求
        """
        try:
            success = self.process_rotation_manager.request_sm_reallocation(new_allocation)
            
            if success:
                self.stats["resource_adjustments"] += 1
                logger.info(f"Resource adjustment requested: P:{new_allocation.prefill_percentage}% D:{new_allocation.decode_percentage}%")
            else:
                logger.warning("Failed to request resource adjustment")
                
            return success
            
        except Exception as e:
            logger.error(f"Error requesting resource adjustment: {e}")
            return False
            
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        return self.storage_manager.get_memory_usage()
        
    def get_slo_compliance(self) -> Dict:
        """获取SLO合规情况"""
        return self.slo_manager.get_slo_compliance_rate()
        
    def force_garbage_collection(self):
        """强制垃圾回收"""
        try:
            self.storage_manager.force_gc()
            logger.info("Forced garbage collection completed")
        except Exception as e:
            logger.error(f"Error during forced garbage collection: {e}")
            
    def cleanup_expired_resources(self, max_age_seconds: float = 3600.0):
        """清理过期资源"""
        try:
            self.storage_manager.cleanup_expired_blocks(max_age_seconds)
            logger.info(f"Cleaned up resources older than {max_age_seconds}s")
        except Exception as e:
            logger.error(f"Error cleaning up expired resources: {e}")
            
    def get_comprehensive_status(self) -> Dict:
        """获取综合状态信息"""
        try:
            uptime = time.time() - self.start_time if self.start_time else 0
            
            status = {
                "coordinator": {
                    "running": self.running,
                    "uptime_seconds": uptime,
                    "stats": self.stats.copy(),
                },
                "process_rotation": self.process_rotation_manager.get_status(),
                "storage_management": self.storage_manager.get_status(),
                "slo_management": self.slo_manager.get_status(),
                "memory_usage": self.get_memory_usage(),
                "slo_compliance": self.get_slo_compliance(),
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return {"error": str(e)}
            
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            health = {
                "coordinator_running": self.running,
                "components_healthy": True,
                "issues": [],
            }
            
            # 检查进程轮换管理器
            rotation_status = self.process_rotation_manager.get_status()
            if not rotation_status.get("active_processes"):
                health["components_healthy"] = False
                health["issues"].append("No active processes in rotation manager")
                
            # 检查存储管理器
            memory_usage = self.get_memory_usage()
            if memory_usage.get("utilization", 0) > 0.95:
                health["issues"].append("Memory utilization too high (>95%)")
                
            # 检查SLO合规性
            slo_compliance = self.get_slo_compliance()
            if slo_compliance.get("overall_compliance", 1.0) < 0.8:
                health["issues"].append("SLO compliance too low (<80%)")
                
            health["healthy"] = health["components_healthy"] and len(health["issues"]) == 0
            
            return health
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "healthy": False,
                "coordinator_running": False,
                "components_healthy": False,
                "issues": [f"Health check failed: {e}"],
            }
            
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# 便利函数
def create_semi_pd_coordinator(
    server_args: ServerArgs,
    port_args: PortArgs,
    prefill_sm_percentage: int = 70,
    decode_sm_percentage: int = 30,
    ttft_target_ms: float = 100.0,
    tpot_target_ms: float = 50.0,
    total_kv_blocks: int = 10000,
    kv_block_size: int = 4096,
    device: Optional[torch.device] = None,
) -> SemiPDCoordinator:
    """
    创建Semi-PD协调器的便利函数
    
    Args:
        server_args: 服务器参数
        port_args: 端口参数
        prefill_sm_percentage: Prefill SM百分比
        decode_sm_percentage: Decode SM百分比
        ttft_target_ms: TTFT目标延迟（毫秒）
        tpot_target_ms: TPOT目标延迟（毫秒）
        total_kv_blocks: KV cache总块数
        kv_block_size: KV cache块大小
        device: 设备
        
    Returns:
        Semi-PD协调器实例
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    initial_allocation = SMAllocation(
        prefill_percentage=prefill_sm_percentage,
        decode_percentage=decode_sm_percentage,
    )
    
    slo_target = SLOTarget(
        ttft_target_ms=ttft_target_ms,
        tpot_target_ms=tpot_target_ms,
    )
    
    return SemiPDCoordinator(
        server_args=server_args,
        port_args=port_args,
        initial_sm_allocation=initial_allocation,
        slo_target=slo_target,
        total_kv_blocks=total_kv_blocks,
        kv_block_size=kv_block_size,
        device=device,
    )