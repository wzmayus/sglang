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
SLO-aware Dynamic Resource Adjustment Algorithm

根据论文第5节实现SLO感知的动态资源调整算法。
论文链接: https://arxiv.org/html/2504.19867v1#S5
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sglang.semi_pd.metrics_collector import AggregatedMetrics
from sglang.semi_pd.process_rotation_manager import SMAllocation

logger = logging.getLogger(__name__)


@dataclass
class SLOConstraints:
    """SLO约束配置"""
    ttft_target_ms: float  # TTFT目标延迟（毫秒）
    tpot_target_ms: float  # TPOT目标延迟（毫秒）
    ttft_percentile: float = 95.0  # TTFT百分位数
    tpot_percentile: float = 95.0  # TPOT百分位数
    
    # 违反阈值
    ttft_violation_threshold: float = 1.1  # 超过目标10%算违反
    tpot_violation_threshold: float = 1.1  # 超过目标10%算违反


@dataclass
class ResourceAllocationCandidate:
    """资源分配候选方案"""
    prefill_percentage: int
    decode_percentage: int
    
    # 预测性能
    predicted_ttft: float = 0.0
    predicted_tpot: float = 0.0
    predicted_throughput: float = 0.0
    
    # 评估得分
    slo_violation_score: float = float('inf')
    throughput_score: float = 0.0
    overall_score: float = float('inf')
    
    @property
    def allocation(self) -> SMAllocation:
        return SMAllocation(
            prefill_percentage=self.prefill_percentage,
            decode_percentage=self.decode_percentage
        )


class SLOAwareAlgorithm:
    """
    SLO感知的动态资源调整算法
    
    根据论文第5节实现，主要功能：
    1. 监控SLO违反情况
    2. 预测不同资源配置下的性能
    3. 选择最优的资源分配方案
    4. 最大化在SLO约束下的吞吐量
    """
    
    def __init__(
        self,
        slo_constraints: SLOConstraints,
        min_prefill_percentage: int = 20,
        max_prefill_percentage: int = 80,
        adjustment_step: int = 5,
        prediction_confidence_threshold: float = 0.7,
    ):
        self.slo_constraints = slo_constraints
        self.min_prefill_percentage = min_prefill_percentage
        self.max_prefill_percentage = max_prefill_percentage
        self.adjustment_step = adjustment_step
        self.prediction_confidence_threshold = prediction_confidence_threshold
        
        # 历史数据用于预测
        self.performance_history: List[Dict] = []
        self.max_history_size = 100
        
        # 算法状态
        self.last_adjustment_time = 0.0
        self.adjustment_cooldown = 60.0  # 60秒冷却期
        
        logger.info("SLO-aware algorithm initialized")
        
    def should_adjust_resources(
        self, 
        current_metrics: AggregatedMetrics,
        current_allocation: SMAllocation
    ) -> bool:
        """
        判断是否需要调整资源
        
        Args:
            current_metrics: 当前性能指标
            current_allocation: 当前资源分配
            
        Returns:
            bool: 是否需要调整
        """
        # 检查冷却期
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return False
            
        # 检查SLO违反
        slo_violations = self._detect_slo_violations(current_metrics)
        
        if slo_violations:
            logger.info(f"SLO violations detected: {slo_violations}")
            return True
            
        # 检查是否有优化空间（即使没有违反SLO）
        optimization_potential = self._assess_optimization_potential(
            current_metrics, current_allocation
        )
        
        if optimization_potential > 0.1:  # 10%以上的优化潜力
            logger.info(f"Optimization potential detected: {optimization_potential:.2%}")
            return True
            
        return False
        
    def compute_optimal_allocation(
        self,
        current_metrics: AggregatedMetrics,
        current_allocation: SMAllocation
    ) -> Optional[SMAllocation]:
        """
        计算最优资源分配
        
        根据论文第5节的算法，在SLO约束下最大化吞吐量
        
        Args:
            current_metrics: 当前性能指标
            current_allocation: 当前资源分配
            
        Returns:
            最优资源分配方案，如果无需调整则返回None
        """
        logger.info("Computing optimal resource allocation")
        
        # 生成候选分配方案
        candidates = self._generate_allocation_candidates(current_allocation)
        
        # 为每个候选方案预测性能
        for candidate in candidates:
            self._predict_performance(candidate, current_metrics)
            
        # 评估候选方案
        valid_candidates = []
        for candidate in candidates:
            if self._evaluate_candidate(candidate):
                valid_candidates.append(candidate)
                
        if not valid_candidates:
            logger.warning("No valid allocation candidates found")
            return None
            
        # 选择最优方案
        best_candidate = self._select_best_candidate(valid_candidates)
        
        if best_candidate:
            logger.info(
                f"Optimal allocation found: P:{best_candidate.prefill_percentage}% "
                f"D:{best_candidate.decode_percentage}% "
                f"(predicted TTFT: {best_candidate.predicted_ttft:.1f}ms, "
                f"predicted TPOT: {best_candidate.predicted_tpot:.1f}ms)"
            )
            
            self.last_adjustment_time = time.time()
            return best_candidate.allocation
        else:
            logger.info("Current allocation is already optimal")
            return None
            
    def _detect_slo_violations(self, metrics: AggregatedMetrics) -> Dict[str, float]:
        """检测SLO违反"""
        violations = {}
        
        # 检查TTFT违反
        if metrics.ttft_p95 > 0:
            ttft_ratio = metrics.ttft_p95 / self.slo_constraints.ttft_target_ms
            if ttft_ratio > self.slo_constraints.ttft_violation_threshold:
                violations["ttft"] = ttft_ratio
                
        # 检查TPOT违反
        if metrics.tpot_p95 > 0:
            tpot_ratio = metrics.tpot_p95 / self.slo_constraints.tpot_target_ms
            if tpot_ratio > self.slo_constraints.tpot_violation_threshold:
                violations["tpot"] = tpot_ratio
                
        return violations
        
    def _assess_optimization_potential(
        self, 
        metrics: AggregatedMetrics, 
        allocation: SMAllocation
    ) -> float:
        """评估优化潜力"""
        # 如果当前性能远好于SLO要求，可能有优化空间
        ttft_margin = 0.0
        if metrics.ttft_p95 > 0:
            ttft_margin = 1.0 - (metrics.ttft_p95 / self.slo_constraints.ttft_target_ms)
            
        tpot_margin = 0.0
        if metrics.tpot_p95 > 0:
            tpot_margin = 1.0 - (metrics.tpot_p95 / self.slo_constraints.tpot_target_ms)
            
        # 如果两个指标都有较大余量，说明有优化潜力
        return min(ttft_margin, tpot_margin)
        
    def _generate_allocation_candidates(
        self, 
        current_allocation: SMAllocation
    ) -> List[ResourceAllocationCandidate]:
        """生成资源分配候选方案"""
        candidates = []
        
        current_prefill = current_allocation.prefill_percentage
        current_decode = current_allocation.decode_percentage
        
        # 生成邻近的候选方案
        for prefill_delta in [-2, -1, 0, 1, 2]:
            for decode_delta in [-2, -1, 0, 1, 2]:
                if prefill_delta == 0 and decode_delta == 0:
                    continue  # 跳过当前配置
                    
                new_prefill = current_prefill + prefill_delta * self.adjustment_step
                new_decode = current_decode + decode_delta * self.adjustment_step
                
                # 检查约束
                if (self.min_prefill_percentage <= new_prefill <= self.max_prefill_percentage and
                    20 <= new_decode <= 80 and
                    new_prefill + new_decode <= 100):
                    
                    candidates.append(ResourceAllocationCandidate(
                        prefill_percentage=new_prefill,
                        decode_percentage=new_decode
                    ))
                    
        # 添加一些极端情况的候选方案
        extreme_candidates = [
            (30, 70), (40, 60), (50, 50), (60, 40), (70, 30)
        ]
        
        for prefill_pct, decode_pct in extreme_candidates:
            if (prefill_pct != current_prefill or decode_pct != current_decode):
                candidates.append(ResourceAllocationCandidate(
                    prefill_percentage=prefill_pct,
                    decode_percentage=decode_pct
                ))
                
        return candidates
        
    def _predict_performance(
        self, 
        candidate: ResourceAllocationCandidate, 
        current_metrics: AggregatedMetrics
    ) -> None:
        """预测候选方案的性能"""
        # 基于历史数据和当前指标进行性能预测
        # 这里使用简化的线性模型，实际实现中可以使用更复杂的机器学习模型
        
        # 基准性能（当前指标）
        base_ttft = current_metrics.ttft_p95 if current_metrics.ttft_p95 > 0 else self.slo_constraints.ttft_target_ms
        base_tpot = current_metrics.tpot_p95 if current_metrics.tpot_p95 > 0 else self.slo_constraints.tpot_target_ms
        base_throughput = current_metrics.request_throughput
        
        # 资源分配对性能的影响因子
        prefill_factor = candidate.prefill_percentage / 50.0  # 归一化到50%基准
        decode_factor = candidate.decode_percentage / 50.0
        
        # 预测TTFT（更多prefill资源 -> 更好的TTFT）
        ttft_improvement = min(2.0, max(0.5, prefill_factor))
        candidate.predicted_ttft = base_ttft / ttft_improvement
        
        # 预测TPOT（更多decode资源 -> 更好的TPOT）
        tpot_improvement = min(2.0, max(0.5, decode_factor))
        candidate.predicted_tpot = base_tpot / tpot_improvement
        
        # 预测吞吐量（平衡的资源分配通常有更好的吞吐量）
        balance_factor = 1.0 - abs(candidate.prefill_percentage - candidate.decode_percentage) / 100.0
        throughput_factor = 0.8 + 0.4 * balance_factor  # 0.8 到 1.2 的范围
        candidate.predicted_throughput = base_throughput * throughput_factor
        
    def _evaluate_candidate(self, candidate: ResourceAllocationCandidate) -> bool:
        """评估候选方案是否有效"""
        # 检查是否满足SLO约束
        ttft_satisfies_slo = candidate.predicted_ttft <= self.slo_constraints.ttft_target_ms
        tpot_satisfies_slo = candidate.predicted_tpot <= self.slo_constraints.tpot_target_ms
        
        if not (ttft_satisfies_slo and tpot_satisfies_slo):
            candidate.slo_violation_score = float('inf')
            candidate.overall_score = float('inf')
            return False
            
        # 计算SLO违反得分（越小越好）
        ttft_score = max(0, candidate.predicted_ttft - self.slo_constraints.ttft_target_ms)
        tpot_score = max(0, candidate.predicted_tpot - self.slo_constraints.tpot_target_ms)
        candidate.slo_violation_score = ttft_score + tpot_score
        
        # 计算吞吐量得分（越大越好）
        candidate.throughput_score = candidate.predicted_throughput
        
        # 计算综合得分（在满足SLO的前提下最大化吞吐量）
        candidate.overall_score = -candidate.throughput_score  # 负号因为我们要最小化得分
        
        return True
        
    def _select_best_candidate(
        self, 
        candidates: List[ResourceAllocationCandidate]
    ) -> Optional[ResourceAllocationCandidate]:
        """选择最佳候选方案"""
        if not candidates:
            return None
            
        # 按综合得分排序（得分越小越好）
        candidates.sort(key=lambda c: c.overall_score)
        
        best_candidate = candidates[0]
        
        # 记录性能历史用于未来预测
        self.performance_history.append({
            "timestamp": time.time(),
            "prefill_percentage": best_candidate.prefill_percentage,
            "decode_percentage": best_candidate.decode_percentage,
            "predicted_ttft": best_candidate.predicted_ttft,
            "predicted_tpot": best_candidate.predicted_tpot,
            "predicted_throughput": best_candidate.predicted_throughput,
        })
        
        # 保持历史记录大小
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
            
        return best_candidate
        
    def update_performance_history(
        self,
        allocation: SMAllocation,
        actual_metrics: AggregatedMetrics
    ) -> None:
        """更新性能历史记录"""
        self.performance_history.append({
            "timestamp": time.time(),
            "prefill_percentage": allocation.prefill_percentage,
            "decode_percentage": allocation.decode_percentage,
            "actual_ttft": actual_metrics.ttft_p95,
            "actual_tpot": actual_metrics.tpot_p95,
            "actual_throughput": actual_metrics.request_throughput,
        })
        
        # 保持历史记录大小
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
            
    def get_algorithm_status(self) -> Dict:
        """获取算法状态"""
        return {
            "slo_constraints": {
                "ttft_target_ms": self.slo_constraints.ttft_target_ms,
                "tpot_target_ms": self.slo_constraints.tpot_target_ms,
                "ttft_percentile": self.slo_constraints.ttft_percentile,
                "tpot_percentile": self.slo_constraints.tpot_percentile,
            },
            "algorithm_config": {
                "min_prefill_percentage": self.min_prefill_percentage,
                "max_prefill_percentage": self.max_prefill_percentage,
                "adjustment_step": self.adjustment_step,
                "prediction_confidence_threshold": self.prediction_confidence_threshold,
                "adjustment_cooldown": self.adjustment_cooldown,
            },
            "runtime_status": {
                "last_adjustment_time": self.last_adjustment_time,
                "performance_history_size": len(self.performance_history),
                "cooldown_remaining": max(0, self.adjustment_cooldown - (time.time() - self.last_adjustment_time)),
            },
        }


class SLOAwareResourceController:
    """
    SLO感知资源控制器
    
    整合metrics收集和SLO算法，提供完整的动态资源调整功能
    """
    
    def __init__(
        self,
        slo_constraints: SLOConstraints,
        metrics_collector,
        process_rotation_manager,
        monitoring_interval: float = 10.0,
    ):
        self.slo_constraints = slo_constraints
        self.metrics_collector = metrics_collector
        self.process_rotation_manager = process_rotation_manager
        self.monitoring_interval = monitoring_interval
        
        # 初始化SLO算法
        self.slo_algorithm = SLOAwareAlgorithm(slo_constraints)
        
        # 控制状态
        self.running = False
        self.monitor_thread = None
        
        # 统计信息
        self.total_adjustments = 0
        self.successful_adjustments = 0
        self.slo_violations_detected = 0
        
        logger.info("SLO-aware resource controller initialized")
        
    def start_monitoring(self) -> None:
        """开始SLO监控"""
        if self.running:
            logger.warning("SLO monitoring is already running")
            return
            
        self.running = True
        
        import threading
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("SLO monitoring started")
        
    def stop_monitoring(self) -> None:
        """停止SLO监控"""
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            
        logger.info("SLO monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.running:
            try:
                # 获取当前指标
                current_metrics = self.metrics_collector.get_current_window_metrics()
                
                # 获取当前资源分配
                status = self.process_rotation_manager.get_status()
                current_allocation = SMAllocation(
                    prefill_percentage=status["current_sm_allocation"]["prefill_percentage"],
                    decode_percentage=status["current_sm_allocation"]["decode_percentage"]
                )
                
                # 检查是否需要调整
                if self.slo_algorithm.should_adjust_resources(current_metrics, current_allocation):
                    self.slo_violations_detected += 1
                    
                    # 计算最优分配
                    optimal_allocation = self.slo_algorithm.compute_optimal_allocation(
                        current_metrics, current_allocation
                    )
                    
                    if optimal_allocation:
                        # 请求资源调整
                        success = self.process_rotation_manager.request_sm_reallocation(optimal_allocation)
                        
                        self.total_adjustments += 1
                        if success:
                            self.successful_adjustments += 1
                            logger.info(f"Resource adjustment requested successfully")
                        else:
                            logger.warning(f"Resource adjustment request failed")
                            
                # 更新性能历史
                self.slo_algorithm.update_performance_history(current_allocation, current_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in SLO monitoring loop: {e}")
                time.sleep(1.0)
                
    def get_controller_status(self) -> Dict:
        """获取控制器状态"""
        return {
            "running": self.running,
            "monitoring_interval": self.monitoring_interval,
            "statistics": {
                "total_adjustments": self.total_adjustments,
                "successful_adjustments": self.successful_adjustments,
                "slo_violations_detected": self.slo_violations_detected,
                "success_rate": (
                    self.successful_adjustments / max(1, self.total_adjustments)
                ),
            },
            "algorithm_status": self.slo_algorithm.get_algorithm_status(),
        }