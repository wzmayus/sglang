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
Semi-PD Metrics Collector

收集Semi-PD运行时的关键性能指标，用于SLO感知的动态资源调整算法。
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sglang.semi_pd.utils import InstanceRole

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """单个请求的指标"""
    request_id: str
    arrival_time: float
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    decode_end_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    input_length: int = 0
    output_length: int = 0
    cached_tokens: int = 0
    
    # 计算得出的指标
    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to First Token (毫秒)"""
        if self.first_token_time and self.arrival_time:
            return (self.first_token_time - self.arrival_time) * 1000
        return None
    
    @property
    def tpot_ms(self) -> Optional[float]:
        """Time per Output Token (毫秒)"""
        if (self.completion_time and self.first_token_time and 
            self.output_length > 1):
            decode_time = self.completion_time - self.first_token_time
            return (decode_time / (self.output_length - 1)) * 1000
        return None
    
    @property
    def e2e_latency_ms(self) -> Optional[float]:
        """端到端延迟 (毫秒)"""
        if self.completion_time and self.arrival_time:
            return (self.completion_time - self.arrival_time) * 1000
        return None


@dataclass
class SystemMetrics:
    """系统级指标"""
    timestamp: float
    
    # 队列长度
    prefill_queue_length: int = 0
    decode_queue_length: int = 0
    grammar_queue_length: int = 0
    
    # 运行中的请求数
    prefill_running_requests: int = 0
    decode_running_requests: int = 0
    
    # 资源利用率
    prefill_utilization: float = 0.0
    decode_utilization: float = 0.0
    
    # 吞吐量
    prefill_throughput: float = 0.0  # tokens/sec
    decode_throughput: float = 0.0   # tokens/sec
    
    # 内存使用
    kv_cache_usage: float = 0.0
    token_usage: float = 0.0
    
    # 缓存命中率
    cache_hit_rate: float = 0.0


@dataclass
class AggregatedMetrics:
    """聚合指标"""
    window_start_time: float
    window_end_time: float
    
    # TTFT统计
    ttft_mean: float = 0.0
    ttft_median: float = 0.0
    ttft_p90: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    
    # TPOT统计
    tpot_mean: float = 0.0
    tpot_median: float = 0.0
    tpot_p90: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0
    
    # 端到端延迟统计
    e2e_latency_mean: float = 0.0
    e2e_latency_median: float = 0.0
    e2e_latency_p95: float = 0.0
    
    # 吞吐量统计
    request_throughput: float = 0.0
    input_throughput: float = 0.0
    output_throughput: float = 0.0
    
    # 系统指标平均值
    avg_prefill_queue_length: float = 0.0
    avg_decode_queue_length: float = 0.0
    avg_prefill_utilization: float = 0.0
    avg_decode_utilization: float = 0.0
    avg_cache_hit_rate: float = 0.0
    
    # 请求统计
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0


class SemiPDMetricsCollector:
    """
    Semi-PD指标收集器
    
    收集运行时的关键性能指标，包括：
    1. 请求级指标：TTFT、TPOT、端到端延迟
    2. 系统级指标：队列长度、利用率、吞吐量
    3. 聚合指标：各种百分位数统计
    """
    
    def __init__(
        self,
        window_size_seconds: float = 30.0,
        max_history_size: int = 1000,
    ):
        self.window_size_seconds = window_size_seconds
        self.max_history_size = max_history_size
        
        # 数据存储
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.system_metrics_history: deque = deque(maxlen=max_history_size)
        self.completed_requests: deque = deque(maxlen=max_history_size)
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 统计信息
        self.total_requests = 0
        self.completed_requests_count = 0
        self.failed_requests_count = 0
        
        logger.info("Semi-PD metrics collector initialized")
        
    def record_request_arrival(self, request_id: str, input_length: int) -> None:
        """记录请求到达"""
        with self.lock:
            current_time = time.time()
            
            self.request_metrics[request_id] = RequestMetrics(
                request_id=request_id,
                arrival_time=current_time,
                input_length=input_length,
            )
            
            self.total_requests += 1
            
    def record_prefill_start(self, request_id: str) -> None:
        """记录prefill开始"""
        with self.lock:
            if request_id in self.request_metrics:
                self.request_metrics[request_id].prefill_start_time = time.time()
                
    def record_prefill_end(self, request_id: str) -> None:
        """记录prefill结束"""
        with self.lock:
            if request_id in self.request_metrics:
                self.request_metrics[request_id].prefill_end_time = time.time()
                
    def record_first_token(self, request_id: str) -> None:
        """记录第一个token生成"""
        with self.lock:
            if request_id in self.request_metrics:
                self.request_metrics[request_id].first_token_time = time.time()
                
    def record_decode_start(self, request_id: str) -> None:
        """记录decode开始"""
        with self.lock:
            if request_id in self.request_metrics:
                self.request_metrics[request_id].decode_start_time = time.time()
                
    def record_request_completion(
        self, 
        request_id: str, 
        output_length: int,
        cached_tokens: int = 0,
        success: bool = True
    ) -> None:
        """记录请求完成"""
        with self.lock:
            current_time = time.time()
            
            if request_id in self.request_metrics:
                metrics = self.request_metrics[request_id]
                metrics.completion_time = current_time
                metrics.output_length = output_length
                metrics.cached_tokens = cached_tokens
                
                if success:
                    self.completed_requests.append(metrics)
                    self.completed_requests_count += 1
                else:
                    self.failed_requests_count += 1
                    
                # 清理已完成的请求指标
                del self.request_metrics[request_id]
                
    def record_system_metrics(self, metrics: SystemMetrics) -> None:
        """记录系统指标"""
        with self.lock:
            self.system_metrics_history.append(metrics)
            
    def get_current_window_metrics(self) -> AggregatedMetrics:
        """获取当前窗口的聚合指标"""
        with self.lock:
            current_time = time.time()
            window_start = current_time - self.window_size_seconds
            
            # 过滤窗口内的完成请求
            window_requests = [
                req for req in self.completed_requests
                if req.completion_time and req.completion_time >= window_start
            ]
            
            # 过滤窗口内的系统指标
            window_system_metrics = [
                metrics for metrics in self.system_metrics_history
                if metrics.timestamp >= window_start
            ]
            
            return self._calculate_aggregated_metrics(
                window_requests, 
                window_system_metrics,
                window_start,
                current_time
            )
            
    def _calculate_aggregated_metrics(
        self,
        requests: List[RequestMetrics],
        system_metrics: List[SystemMetrics],
        window_start: float,
        window_end: float
    ) -> AggregatedMetrics:
        """计算聚合指标"""
        
        aggregated = AggregatedMetrics(
            window_start_time=window_start,
            window_end_time=window_end,
        )
        
        if not requests:
            return aggregated
            
        # 计算TTFT统计
        ttfts = [req.ttft_ms for req in requests if req.ttft_ms is not None]
        if ttfts:
            aggregated.ttft_mean = np.mean(ttfts)
            aggregated.ttft_median = np.median(ttfts)
            aggregated.ttft_p90 = np.percentile(ttfts, 90)
            aggregated.ttft_p95 = np.percentile(ttfts, 95)
            aggregated.ttft_p99 = np.percentile(ttfts, 99)
            
        # 计算TPOT统计
        tpots = [req.tpot_ms for req in requests if req.tpot_ms is not None]
        if tpots:
            aggregated.tpot_mean = np.mean(tpots)
            aggregated.tpot_median = np.median(tpots)
            aggregated.tpot_p90 = np.percentile(tpots, 90)
            aggregated.tpot_p95 = np.percentile(tpots, 95)
            aggregated.tpot_p99 = np.percentile(tpots, 99)
            
        # 计算端到端延迟统计
        e2e_latencies = [req.e2e_latency_ms for req in requests if req.e2e_latency_ms is not None]
        if e2e_latencies:
            aggregated.e2e_latency_mean = np.mean(e2e_latencies)
            aggregated.e2e_latency_median = np.median(e2e_latencies)
            aggregated.e2e_latency_p95 = np.percentile(e2e_latencies, 95)
            
        # 计算吞吐量
        window_duration = window_end - window_start
        if window_duration > 0:
            aggregated.total_requests = len(requests)
            aggregated.completed_requests = len([r for r in requests if r.completion_time])
            aggregated.request_throughput = aggregated.completed_requests / window_duration
            
            total_input_tokens = sum(req.input_length for req in requests)
            total_output_tokens = sum(req.output_length for req in requests)
            
            aggregated.input_throughput = total_input_tokens / window_duration
            aggregated.output_throughput = total_output_tokens / window_duration
            
        # 计算系统指标平均值
        if system_metrics:
            aggregated.avg_prefill_queue_length = np.mean([m.prefill_queue_length for m in system_metrics])
            aggregated.avg_decode_queue_length = np.mean([m.decode_queue_length for m in system_metrics])
            aggregated.avg_prefill_utilization = np.mean([m.prefill_utilization for m in system_metrics])
            aggregated.avg_decode_utilization = np.mean([m.decode_utilization for m in system_metrics])
            aggregated.avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in system_metrics])
            
        return aggregated
        
    def get_real_time_metrics(self) -> Dict:
        """获取实时指标摘要"""
        with self.lock:
            current_metrics = self.get_current_window_metrics()
            
            # 获取最新的系统指标
            latest_system_metrics = None
            if self.system_metrics_history:
                latest_system_metrics = self.system_metrics_history[-1]
                
            return {
                "timestamp": time.time(),
                "window_size_seconds": self.window_size_seconds,
                
                # 延迟指标
                "ttft_p95_ms": current_metrics.ttft_p95,
                "tpot_p95_ms": current_metrics.tpot_p95,
                "e2e_latency_p95_ms": current_metrics.e2e_latency_p95,
                
                # 吞吐量指标
                "request_throughput": current_metrics.request_throughput,
                "input_throughput": current_metrics.input_throughput,
                "output_throughput": current_metrics.output_throughput,
                
                # 队列和利用率
                "prefill_queue_length": latest_system_metrics.prefill_queue_length if latest_system_metrics else 0,
                "decode_queue_length": latest_system_metrics.decode_queue_length if latest_system_metrics else 0,
                "prefill_utilization": latest_system_metrics.prefill_utilization if latest_system_metrics else 0.0,
                "decode_utilization": latest_system_metrics.decode_utilization if latest_system_metrics else 0.0,
                
                # 统计信息
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests_count,
                "failed_requests": self.failed_requests_count,
                "active_requests": len(self.request_metrics),
                
                # 缓存命中率
                "cache_hit_rate": latest_system_metrics.cache_hit_rate if latest_system_metrics else 0.0,
            }
            
    def export_metrics_for_slo_algorithm(self) -> Dict:
        """导出用于SLO算法的指标"""
        current_metrics = self.get_current_window_metrics()
        
        return {
            # SLO关键指标
            "ttft_p95": current_metrics.ttft_p95,
            "tpot_p95": current_metrics.tpot_p95,
            
            # 负载指标
            "prefill_queue_length": current_metrics.avg_prefill_queue_length,
            "decode_queue_length": current_metrics.avg_decode_queue_length,
            "prefill_utilization": current_metrics.avg_prefill_utilization,
            "decode_utilization": current_metrics.avg_decode_utilization,
            
            # 吞吐量指标
            "prefill_throughput": current_metrics.input_throughput,
            "decode_throughput": current_metrics.output_throughput,
            
            # 时间戳
            "timestamp": time.time(),
            "window_duration": self.window_size_seconds,
        }
        
    def reset_statistics(self) -> None:
        """重置统计信息"""
        with self.lock:
            self.request_metrics.clear()
            self.system_metrics_history.clear()
            self.completed_requests.clear()
            
            self.total_requests = 0
            self.completed_requests_count = 0
            self.failed_requests_count = 0
            
            logger.info("Semi-PD metrics statistics reset")
            
    def get_statistics_summary(self) -> Dict:
        """获取统计摘要"""
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests_count,
                "failed_requests": self.failed_requests_count,
                "active_requests": len(self.request_metrics),
                "success_rate": (
                    self.completed_requests_count / max(1, self.total_requests)
                ),
                "system_metrics_count": len(self.system_metrics_history),
                "window_size_seconds": self.window_size_seconds,
            }