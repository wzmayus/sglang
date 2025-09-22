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
Semi-PD Metrics Integration

将Semi-PD的metrics收集器与SGLang现有的scheduler集成，
实现运行时性能指标的自动收集。
"""

import logging
import time
from typing import Dict, List, Optional

from sglang.semi_pd.metrics_collector import (
    SemiPDMetricsCollector,
    SystemMetrics,
    RequestMetrics,
)
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.metrics.collector import SchedulerStats

logger = logging.getLogger(__name__)


class SemiPDMetricsIntegration:
    """
    Semi-PD Metrics集成器
    
    负责：
    1. 从SGLang scheduler收集运行时指标
    2. 转换为Semi-PD格式的指标
    3. 提供给SLO算法使用
    """
    
    def __init__(
        self,
        instance_role: InstanceRole,
        window_size_seconds: float = 30.0,
    ):
        self.instance_role = instance_role
        self.metrics_collector = SemiPDMetricsCollector(
            window_size_seconds=window_size_seconds
        )
        
        # 请求跟踪
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        
        logger.info(f"Semi-PD metrics integration initialized for {instance_role.name}")
        
    def on_request_arrival(self, req: Req) -> None:
        """处理请求到达事件"""
        try:
            self.metrics_collector.record_request_arrival(
                request_id=req.rid,
                input_length=len(req.origin_input_ids) if req.origin_input_ids else 0
            )
            
            self.active_requests[req.rid] = time.time()
            
        except Exception as e:
            logger.error(f"Error recording request arrival: {e}")
            
    def on_prefill_start(self, batch: ScheduleBatch) -> None:
        """处理prefill开始事件"""
        if self.instance_role != InstanceRole.PREFILL:
            return
            
        try:
            for req in batch.reqs:
                self.metrics_collector.record_prefill_start(req.rid)
                
        except Exception as e:
            logger.error(f"Error recording prefill start: {e}")
            
    def on_prefill_end(self, batch: ScheduleBatch) -> None:
        """处理prefill结束事件"""
        if self.instance_role != InstanceRole.PREFILL:
            return
            
        try:
            for req in batch.reqs:
                self.metrics_collector.record_prefill_end(req.rid)
                
        except Exception as e:
            logger.error(f"Error recording prefill end: {e}")
            
    def on_first_token_generated(self, req: Req) -> None:
        """处理第一个token生成事件"""
        try:
            self.metrics_collector.record_first_token(req.rid)
            
        except Exception as e:
            logger.error(f"Error recording first token: {e}")
            
    def on_decode_start(self, batch: ScheduleBatch) -> None:
        """处理decode开始事件"""
        if self.instance_role != InstanceRole.DECODE:
            return
            
        try:
            for req in batch.reqs:
                self.metrics_collector.record_decode_start(req.rid)
                
        except Exception as e:
            logger.error(f"Error recording decode start: {e}")
            
    def on_request_completion(
        self, 
        req: Req, 
        success: bool = True,
        cached_tokens: int = 0
    ) -> None:
        """处理请求完成事件"""
        try:
            output_length = len(req.output_ids) if req.output_ids else 0
            
            self.metrics_collector.record_request_completion(
                request_id=req.rid,
                output_length=output_length,
                cached_tokens=cached_tokens,
                success=success
            )
            
            # 清理活跃请求记录
            if req.rid in self.active_requests:
                del self.active_requests[req.rid]
                
        except Exception as e:
            logger.error(f"Error recording request completion: {e}")
            
    def update_system_metrics(self, scheduler_stats: SchedulerStats) -> None:
        """从scheduler stats更新系统指标"""
        try:
            current_time = time.time()
            
            # 构建系统指标
            system_metrics = SystemMetrics(
                timestamp=current_time,
                prefill_queue_length=scheduler_stats.num_queue_reqs if self.instance_role == InstanceRole.PREFILL else 0,
                decode_queue_length=scheduler_stats.num_queue_reqs if self.instance_role == InstanceRole.DECODE else 0,
                grammar_queue_length=scheduler_stats.num_grammar_queue_reqs,
                prefill_running_requests=scheduler_stats.num_running_reqs if self.instance_role == InstanceRole.PREFILL else 0,
                decode_running_requests=scheduler_stats.num_running_reqs if self.instance_role == InstanceRole.DECODE else 0,
                prefill_utilization=scheduler_stats.token_usage if self.instance_role == InstanceRole.PREFILL else 0.0,
                decode_utilization=scheduler_stats.token_usage if self.instance_role == InstanceRole.DECODE else 0.0,
                prefill_throughput=scheduler_stats.gen_throughput if self.instance_role == InstanceRole.PREFILL else 0.0,
                decode_throughput=scheduler_stats.gen_throughput if self.instance_role == InstanceRole.DECODE else 0.0,
                kv_cache_usage=scheduler_stats.token_usage,
                token_usage=scheduler_stats.token_usage,
                cache_hit_rate=scheduler_stats.cache_hit_rate,
            )
            
            self.metrics_collector.record_system_metrics(system_metrics)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            
    def get_metrics_for_slo_algorithm(self) -> Dict:
        """获取用于SLO算法的指标"""
        return self.metrics_collector.export_metrics_for_slo_algorithm()
        
    def get_real_time_metrics(self) -> Dict:
        """获取实时指标"""
        return self.metrics_collector.get_real_time_metrics()
        
    def get_aggregated_metrics(self):
        """获取聚合指标"""
        return self.metrics_collector.get_current_window_metrics()


class SemiPDSchedulerMetricsMixin:
    """
    Semi-PD Scheduler Metrics Mixin
    
    扩展现有的scheduler，添加Semi-PD指标收集功能
    """
    
    def init_semi_pd_metrics(self, instance_role: InstanceRole):
        """初始化Semi-PD指标收集"""
        self.semi_pd_metrics = SemiPDMetricsIntegration(
            instance_role=instance_role,
            window_size_seconds=30.0
        )
        
        logger.info(f"Semi-PD metrics integration enabled for {instance_role.name}")
        
    def semi_pd_on_request_arrival(self, req: Req):
        """Semi-PD: 处理请求到达"""
        if hasattr(self, 'semi_pd_metrics'):
            self.semi_pd_metrics.on_request_arrival(req)
            
    def semi_pd_on_batch_prefill_start(self, batch: ScheduleBatch):
        """Semi-PD: 处理prefill批次开始"""
        if hasattr(self, 'semi_pd_metrics'):
            self.semi_pd_metrics.on_prefill_start(batch)
            
    def semi_pd_on_batch_prefill_end(self, batch: ScheduleBatch):
        """Semi-PD: 处理prefill批次结束"""
        if hasattr(self, 'semi_pd_metrics'):
            self.semi_pd_metrics.on_prefill_end(batch)
            
    def semi_pd_on_first_token_generated(self, req: Req):
        """Semi-PD: 处理第一个token生成"""
        if hasattr(self, 'semi_pd_metrics'):
            self.semi_pd_metrics.on_first_token_generated(req)
            
    def semi_pd_on_batch_decode_start(self, batch: ScheduleBatch):
        """Semi-PD: 处理decode批次开始"""
        if hasattr(self, 'semi_pd_metrics'):
            self.semi_pd_metrics.on_decode_start(batch)
            
    def semi_pd_on_request_completion(self, req: Req, success: bool = True):
        """Semi-PD: 处理请求完成"""
        if hasattr(self, 'semi_pd_metrics'):
            # 尝试获取缓存命中的token数
            cached_tokens = 0
            if hasattr(req, 'cached_tokens'):
                cached_tokens = req.cached_tokens
                
            self.semi_pd_metrics.on_request_completion(
                req=req,
                success=success,
                cached_tokens=cached_tokens
            )
            
    def semi_pd_update_system_metrics(self):
        """Semi-PD: 更新系统指标"""
        if hasattr(self, 'semi_pd_metrics') and hasattr(self, 'stats'):
            self.semi_pd_metrics.update_system_metrics(self.stats)
            
    def semi_pd_get_metrics_summary(self) -> Dict:
        """Semi-PD: 获取指标摘要"""
        if hasattr(self, 'semi_pd_metrics'):
            return {
                "real_time_metrics": self.semi_pd_metrics.get_real_time_metrics(),
                "slo_algorithm_metrics": self.semi_pd_metrics.get_metrics_for_slo_algorithm(),
                "aggregated_metrics": self.semi_pd_metrics.get_aggregated_metrics(),
            }
        return {}


def integrate_semi_pd_metrics_with_scheduler(scheduler: Scheduler, instance_role: InstanceRole):
    """
    将Semi-PD指标收集集成到现有的scheduler中
    
    Args:
        scheduler: SGLang scheduler实例
        instance_role: 实例角色（PREFILL或DECODE）
    """
    # 添加mixin方法到scheduler
    for method_name in dir(SemiPDSchedulerMetricsMixin):
        if method_name.startswith('semi_pd_') or method_name.startswith('init_semi_pd_'):
            method = getattr(SemiPDSchedulerMetricsMixin, method_name)
            setattr(scheduler, method_name, method.__get__(scheduler, type(scheduler)))
    
    # 初始化Semi-PD指标收集
    scheduler.init_semi_pd_metrics(instance_role)
    
    logger.info(f"Semi-PD metrics integration completed for {instance_role.name} scheduler")


class MetricsAggregator:
    """
    跨进程的指标聚合器
    
    在Semi-PD架构中，prefill和decode运行在不同进程中，
    需要聚合两个进程的指标来获得完整的系统视图。
    """
    
    def __init__(self):
        self.prefill_metrics: Optional[Dict] = None
        self.decode_metrics: Optional[Dict] = None
        self.last_update_time = time.time()
        
    def update_prefill_metrics(self, metrics: Dict):
        """更新prefill指标"""
        self.prefill_metrics = metrics
        self.last_update_time = time.time()
        
    def update_decode_metrics(self, metrics: Dict):
        """更新decode指标"""
        self.decode_metrics = metrics
        self.last_update_time = time.time()
        
    def get_aggregated_metrics(self) -> Dict:
        """获取聚合后的指标"""
        if not self.prefill_metrics or not self.decode_metrics:
            return {}
            
        # 聚合TTFT和TPOT（主要来自各自的进程）
        ttft_p95 = self.prefill_metrics.get("ttft_p95_ms", 0)
        tpot_p95 = self.decode_metrics.get("tpot_p95_ms", 0)
        
        # 聚合队列长度
        total_queue_length = (
            self.prefill_metrics.get("prefill_queue_length", 0) +
            self.decode_metrics.get("decode_queue_length", 0)
        )
        
        # 聚合吞吐量
        total_throughput = (
            self.prefill_metrics.get("input_throughput", 0) +
            self.decode_metrics.get("output_throughput", 0)
        )
        
        # 聚合利用率
        avg_utilization = (
            self.prefill_metrics.get("prefill_utilization", 0) +
            self.decode_metrics.get("decode_utilization", 0)
        ) / 2.0
        
        return {
            "timestamp": self.last_update_time,
            "ttft_p95_ms": ttft_p95,
            "tpot_p95_ms": tpot_p95,
            "total_queue_length": total_queue_length,
            "prefill_queue_length": self.prefill_metrics.get("prefill_queue_length", 0),
            "decode_queue_length": self.decode_metrics.get("decode_queue_length", 0),
            "prefill_utilization": self.prefill_metrics.get("prefill_utilization", 0),
            "decode_utilization": self.decode_metrics.get("decode_utilization", 0),
            "avg_utilization": avg_utilization,
            "total_throughput": total_throughput,
            "prefill_throughput": self.prefill_metrics.get("input_throughput", 0),
            "decode_throughput": self.decode_metrics.get("output_throughput", 0),
            "cache_hit_rate": max(
                self.prefill_metrics.get("cache_hit_rate", 0),
                self.decode_metrics.get("cache_hit_rate", 0)
            ),
        }