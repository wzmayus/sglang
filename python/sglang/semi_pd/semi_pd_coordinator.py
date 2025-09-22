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
"""
Semi-PD Coordinator

整合三个核心功能：
1. 常驻进程+进程轮转机制
2. SLO-aware动态资源调整算法
3. Unified Memory Manager

提供统一的接口供SGLang在线服务使用。
"""

import logging
import threading
import time
from typing import Dict, Optional

import torch

from sglang.semi_pd.process_rotation_manager import ProcessRotationManager, SMAllocation
from sglang.semi_pd.resident_process_manager import ResidentProcessManager
from sglang.semi_pd.unified_memory_manager import UnifiedMemoryManager
from sglang.semi_pd.metrics_collector import SemiPDMetricsCollector
from sglang.semi_pd.metrics_integration import MetricsAggregator
from sglang.semi_pd.slo_algorithm import SLOConstraints, SLOAwareResourceController
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import ServerArgs, SemiPDPortArgs

logger = logging.getLogger(__name__)


class SemiPDCoordinator:
    """
    Semi-PD协调器
    
    整合并协调三个核心功能：
    1. 常驻进程+进程轮转机制
    2. SLO-aware动态资源调整算法  
    3. Unified Memory Manager
    """
    
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: SemiPDPortArgs,
        initial_sm_allocation: SMAllocation,
        slo_constraints: Optional[SLOConstraints] = None,
        unified_memory_config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        self.server_args = server_args
        self.port_args = port_args
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心组件
        self.process_rotation_manager = None
        self.unified_memory_manager = None
        self.slo_controller = None
        self.metrics_aggregator = None
        
        # 配置
        self.initial_sm_allocation = initial_sm_allocation
        self.slo_constraints = slo_constraints
        self.unified_memory_config = unified_memory_config or {}
        
        # 状态
        self.is_running = False
        self.initialization_lock = threading.Lock()
        
        logger.info("Semi-PD Coordinator created")
        
    def initialize(self) -> bool:
        """初始化所有组件"""
        with self.initialization_lock:
            if self.is_running:
                logger.warning("Semi-PD Coordinator is already running")
                return True
                
            try:
                # 1. 初始化Unified Memory Manager
                if self._should_enable_unified_memory():
                    self._init_unified_memory_manager()
                    
                # 2. 初始化进程轮换管理器
                self._init_process_rotation_manager()
                
                # 3. 初始化Metrics聚合器
                self._init_metrics_aggregator()
                
                # 4. 初始化SLO控制器
                if self.slo_constraints:
                    self._init_slo_controller()
                    
                self.is_running = True
                logger.info("Semi-PD Coordinator initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize Semi-PD Coordinator: {e}")
                self._cleanup_partial_initialization()
                return False
                
    def _should_enable_unified_memory(self) -> bool:
        """检查是否应该启用统一内存管理器"""
        return (
            getattr(self.server_args, 'enable_unified_memory', False) or
            self.unified_memory_config.get('enabled', False)
        )
        
    def _init_unified_memory_manager(self):
        """初始化统一内存管理器"""
        config = self.unified_memory_config
        
        self.unified_memory_manager = UnifiedMemoryManager(
            total_blocks=config.get('total_blocks', 1000),
            block_size=config.get('block_size', 4096),
            page_size=config.get('page_size', 16),
            device=self.device,
            dtype=torch.float16,
            enable_prefix_caching=config.get('enable_prefix_caching', False),
        )
        
        logger.info("Unified Memory Manager initialized")
        
    def _init_process_rotation_manager(self):
        """初始化进程轮换管理器"""
        # 如果有统一内存管理器，传递给进程轮换管理器
        rotation_kwargs = {}
        if self.unified_memory_manager:
            rotation_kwargs['unified_memory_manager'] = self.unified_memory_manager
            
        self.process_rotation_manager = ProcessRotationManager(
            server_args=self.server_args,
            port_args=self.port_args,
            initial_sm_allocation=self.initial_sm_allocation,
            gpu_id=getattr(self.server_args, 'base_gpu_id', 0),
            tp_rank=0,
            dp_rank=getattr(self.server_args, 'dp_rank', None),
            **rotation_kwargs
        )
        
        logger.info("Process Rotation Manager initialized")
        
    def _init_metrics_aggregator(self):
        """初始化Metrics聚合器"""
        self.metrics_aggregator = MetricsAggregator()
        logger.info("Metrics Aggregator initialized")
        
    def _init_slo_controller(self):
        """初始化SLO控制器"""
        if not self.metrics_aggregator:
            logger.error("Cannot initialize SLO controller without metrics aggregator")
            return
            
        if not self.process_rotation_manager:
            logger.error("Cannot initialize SLO controller without process rotation manager")
            return
            
        # 创建主要的metrics收集器
        main_metrics_collector = SemiPDMetricsCollector(
            window_size_seconds=getattr(self.server_args, 'slo_window_size', 30.0)
        )
        
        self.slo_controller = SLOAwareResourceController(
            slo_constraints=self.slo_constraints,
            metrics_collector=main_metrics_collector,
            process_rotation_manager=self.process_rotation_manager,
            monitoring_interval=getattr(self.server_args, 'slo_monitoring_interval', 10.0),
        )
        
        logger.info("SLO-aware Resource Controller initialized")
        
    def start(self) -> bool:
        """启动所有组件"""
        if not self.is_running:
            if not self.initialize():
                return False
                
        try:
            # 1. 启动进程轮换管理器
            if self.process_rotation_manager:
                success = self.process_rotation_manager.start()
                if not success:
                    logger.error("Failed to start Process Rotation Manager")
                    return False
                    
            # 2. 启动SLO控制器
            if self.slo_controller:
                self.slo_controller.start_monitoring()
                
            logger.info("Semi-PD Coordinator started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Semi-PD Coordinator: {e}")
            return False
            
    def stop(self):
        """停止所有组件"""
        logger.info("Stopping Semi-PD Coordinator...")
        
        try:
            # 1. 停止SLO控制器
            if self.slo_controller:
                self.slo_controller.stop_monitoring()
                
            # 2. 停止进程轮换管理器
            if self.process_rotation_manager:
                self.process_rotation_manager.stop()
                
            # 3. 清理统一内存管理器
            if self.unified_memory_manager:
                self.unified_memory_manager.force_garbage_collection()
                
            self.is_running = False
            logger.info("Semi-PD Coordinator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Semi-PD Coordinator: {e}")
            
    def _cleanup_partial_initialization(self):
        """清理部分初始化的组件"""
        if self.slo_controller:
            try:
                self.slo_controller.stop_monitoring()
            except:
                pass
            self.slo_controller = None
            
        if self.process_rotation_manager:
            try:
                self.process_rotation_manager.stop()
            except:
                pass
            self.process_rotation_manager = None
            
        if self.unified_memory_manager:
            try:
                self.unified_memory_manager.force_garbage_collection()
            except:
                pass
            self.unified_memory_manager = None
            
        self.metrics_aggregator = None
        
    def get_status(self) -> Dict:
        """获取协调器状态"""
        status = {
            "is_running": self.is_running,
            "components": {
                "process_rotation_manager": self.process_rotation_manager is not None,
                "unified_memory_manager": self.unified_memory_manager is not None,
                "slo_controller": self.slo_controller is not None,
                "metrics_aggregator": self.metrics_aggregator is not None,
            },
            "configuration": {
                "initial_sm_allocation": {
                    "prefill_percentage": self.initial_sm_allocation.prefill_percentage,
                    "decode_percentage": self.initial_sm_allocation.decode_percentage,
                },
                "slo_enabled": self.slo_constraints is not None,
                "unified_memory_enabled": self.unified_memory_manager is not None,
            },
        }
        
        # 添加各组件的详细状态
        if self.process_rotation_manager:
            status["process_rotation_status"] = self.process_rotation_manager.get_status()
            
        if self.unified_memory_manager:
            status["unified_memory_status"] = self.unified_memory_manager.get_status()
            
        if self.slo_controller:
            status["slo_controller_status"] = self.slo_controller.get_controller_status()
            
        if self.metrics_aggregator:
            status["aggregated_metrics"] = self.metrics_aggregator.get_aggregated_metrics()
            
        return status
        
    def update_metrics(self, prefill_metrics: Dict, decode_metrics: Dict):
        """更新聚合指标"""
        if self.metrics_aggregator:
            self.metrics_aggregator.update_prefill_metrics(prefill_metrics)
            self.metrics_aggregator.update_decode_metrics(decode_metrics)
            
    def request_sm_reallocation(self, new_allocation: SMAllocation) -> bool:
        """请求SM重新分配"""
        if self.process_rotation_manager:
            return self.process_rotation_manager.request_sm_reallocation(new_allocation)
        return False
        
    def get_current_sm_allocation(self) -> Optional[SMAllocation]:
        """获取当前SM分配"""
        if self.process_rotation_manager:
            status = self.process_rotation_manager.get_status()
            current_allocation = status.get("current_sm_allocation")
            if current_allocation:
                return SMAllocation(
                    prefill_percentage=current_allocation["prefill_percentage"],
                    decode_percentage=current_allocation["decode_percentage"]
                )
        return None
        
    def get_unified_memory_usage(self) -> Optional[Dict]:
        """获取统一内存使用情况"""
        if self.unified_memory_manager:
            return self.unified_memory_manager.get_memory_usage()
        return None
        
    def allocate_kv_cache_blocks(
        self,
        request_id: str,
        layer_id: int,
        num_blocks: int,
        requester_role: InstanceRole,
    ) -> Optional[list]:
        """分配KV cache块"""
        if self.unified_memory_manager:
            return self.unified_memory_manager.allocate_kv_cache_blocks(
                request_id=request_id,
                layer_id=layer_id,
                num_blocks=num_blocks,
                requester_role=requester_role,
            )
        return None
        
    def deallocate_kv_cache_blocks(self, request_id: str) -> bool:
        """释放KV cache块"""
        if self.unified_memory_manager:
            return self.unified_memory_manager.deallocate_kv_cache_blocks(request_id)
        return True
        
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


def create_semi_pd_coordinator(
    server_args: ServerArgs,
    port_args: SemiPDPortArgs,
    initial_sm_allocation: Optional[SMAllocation] = None,
    slo_constraints: Optional[SLOConstraints] = None,
    unified_memory_config: Optional[Dict] = None,
) -> SemiPDCoordinator:
    """
    创建Semi-PD协调器的便捷函数
    
    Args:
        server_args: 服务器参数
        port_args: 端口参数
        initial_sm_allocation: 初始SM分配
        slo_constraints: SLO约束
        unified_memory_config: 统一内存配置
        
    Returns:
        配置好的Semi-PD协调器
    """
    # 默认SM分配
    if initial_sm_allocation is None:
        initial_sm_allocation = SMAllocation(
            prefill_percentage=getattr(server_args, 'initial_prefill_sm', 70),
            decode_percentage=getattr(server_args, 'initial_decode_sm', 30),
        )
        
    # 默认SLO约束
    if slo_constraints is None and getattr(server_args, 'enable_slo_aware', False):
        slo_constraints = SLOConstraints(
            ttft_target_ms=getattr(server_args, 'slo_ttft_target', 100.0),
            tpot_target_ms=getattr(server_args, 'slo_tpot_target', 50.0),
        )
        
    # 默认统一内存配置
    if unified_memory_config is None:
        unified_memory_config = {
            'enabled': getattr(server_args, 'enable_unified_memory', False),
            'total_blocks': getattr(server_args, 'unified_memory_blocks', 1000),
            'block_size': getattr(server_args, 'unified_memory_block_size', 4096),
            'page_size': getattr(server_args, 'unified_memory_page_size', 16),
            'enable_prefix_caching': getattr(server_args, 'enable_prefix_caching', False),
        }
        
    return SemiPDCoordinator(
        server_args=server_args,
        port_args=port_args,
        initial_sm_allocation=initial_sm_allocation,
        slo_constraints=slo_constraints,
        unified_memory_config=unified_memory_config,
    )