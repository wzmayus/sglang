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
"""SLO-aware dynamic resource adjustment mechanism for Semi-PD."""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sglang.semi_pd.process_rotation_manager import ProcessRotationManager, SMAllocation
from sglang.semi_pd.utils import InstanceRole

logger = logging.getLogger(__name__)


@dataclass
class SLOTarget:
    """SLO目标配置"""
    ttft_target_ms: float  # Time to First Token目标延迟（毫秒）
    tpot_target_ms: float  # Time per Output Token目标延迟（毫秒）
    ttft_percentile: float = 95.0  # TTFT百分位数
    tpot_percentile: float = 95.0  # TPOT百分位数
    

@dataclass
class WorkloadMetrics:
    """工作负载指标"""
    timestamp: float
    prefill_queue_length: int
    decode_queue_length: int
    prefill_throughput: float  # tokens/sec
    decode_throughput: float   # tokens/sec
    ttft_p95: float           # ms
    tpot_p95: float           # ms
    prefill_utilization: float  # 0-1
    decode_utilization: float   # 0-1
    

@dataclass
class ResourcePrediction:
    """资源预测结果"""
    predicted_ttft: float
    predicted_tpot: float
    confidence: float
    recommended_allocation: SMAllocation
    

class SLOAwareResourceManager:
    """
    SLO感知的动态资源调整管理器
    
    功能：
    1. 周期性监控系统负载和性能指标
    2. 检测SLO违反情况
    3. 预测最优SM配比
    4. 触发动态资源调整
    """
    
    def __init__(
        self,
        slo_target: SLOTarget,
        process_rotation_manager: ProcessRotationManager,
        window_size_seconds: float = 30.0,
        adjustment_cooldown_seconds: float = 60.0,
        min_confidence_threshold: float = 0.7,
    ):
        self.slo_target = slo_target
        self.process_rotation_manager = process_rotation_manager
        self.window_size_seconds = window_size_seconds
        self.adjustment_cooldown_seconds = adjustment_cooldown_seconds
        self.min_confidence_threshold = min_confidence_threshold
        
        # 历史数据窗口
        self.metrics_window = deque(maxlen=int(window_size_seconds))
        self.performance_history = deque(maxlen=100)  # 保存更长的历史用于预测
        
        # 控制状态
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_adjustment_time = 0.0
        
        # 预测模型参数
        self.prediction_weights = {
            "queue_length_weight": 0.3,
            "utilization_weight": 0.4,
            "throughput_weight": 0.3,
        }
        
        # SLO违反统计
        self.slo_violations = {
            "ttft_violations": 0,
            "tpot_violations": 0,
            "total_measurements": 0,
        }
        
        logger.info("Initialized SLO-aware resource manager")
        
    def start(self):
        """启动资源管理器"""
        logger.info("Starting SLO-aware resource manager")
        self.running = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """停止资源管理器"""
        logger.info("Stopping SLO-aware resource manager")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
    def report_metrics(self, metrics: WorkloadMetrics):
        """报告工作负载指标"""
        self.metrics_window.append(metrics)
        self.performance_history.append(metrics)
        
        # 更新SLO违反统计
        self._update_slo_violations(metrics)
        
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 检查是否有足够的数据进行分析
                if len(self.metrics_window) < 5:
                    time.sleep(1.0)
                    continue
                    
                # 分析当前窗口的性能
                current_performance = self._analyze_current_performance()
                
                # 检测SLO违反
                slo_violation = self._detect_slo_violation(current_performance)
                
                if slo_violation:
                    logger.warning(f"SLO violation detected: {slo_violation}")
                    
                    # 检查是否可以进行调整（冷却期）
                    current_time = time.time()
                    if current_time - self.last_adjustment_time >= self.adjustment_cooldown_seconds:
                        
                        # 预测最优资源配比
                        prediction = self._predict_optimal_allocation()
                        
                        if prediction and prediction.confidence >= self.min_confidence_threshold:
                            logger.info(
                                f"Requesting resource reallocation: "
                                f"P:{prediction.recommended_allocation.prefill_percentage}% "
                                f"D:{prediction.recommended_allocation.decode_percentage}% "
                                f"(confidence: {prediction.confidence:.2f})"
                            )
                            
                            # 请求资源重新分配
                            success = self.process_rotation_manager.request_sm_reallocation(
                                prediction.recommended_allocation
                            )
                            
                            if success:
                                self.last_adjustment_time = current_time
                                logger.info("Resource reallocation request submitted")
                            else:
                                logger.warning("Failed to submit resource reallocation request")
                        else:
                            logger.info(
                                f"Prediction confidence too low: {prediction.confidence if prediction else 0:.2f}"
                            )
                    else:
                        remaining_cooldown = self.adjustment_cooldown_seconds - (current_time - self.last_adjustment_time)
                        logger.info(f"Resource adjustment in cooldown: {remaining_cooldown:.1f}s remaining")
                        
                time.sleep(5.0)  # 监控间隔
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(1.0)
                
    def _analyze_current_performance(self) -> Dict:
        """分析当前性能"""
        if not self.metrics_window:
            return {}
            
        recent_metrics = list(self.metrics_window)[-10:]  # 最近10个数据点
        
        # 计算平均值
        avg_ttft = np.mean([m.ttft_p95 for m in recent_metrics])
        avg_tpot = np.mean([m.tpot_p95 for m in recent_metrics])
        avg_prefill_util = np.mean([m.prefill_utilization for m in recent_metrics])
        avg_decode_util = np.mean([m.decode_utilization for m in recent_metrics])
        avg_prefill_queue = np.mean([m.prefill_queue_length for m in recent_metrics])
        avg_decode_queue = np.mean([m.decode_queue_length for m in recent_metrics])
        
        # 计算趋势
        ttft_trend = self._calculate_trend([m.ttft_p95 for m in recent_metrics])
        tpot_trend = self._calculate_trend([m.tpot_p95 for m in recent_metrics])
        
        return {
            "avg_ttft": avg_ttft,
            "avg_tpot": avg_tpot,
            "avg_prefill_utilization": avg_prefill_util,
            "avg_decode_utilization": avg_decode_util,
            "avg_prefill_queue": avg_prefill_queue,
            "avg_decode_queue": avg_decode_queue,
            "ttft_trend": ttft_trend,
            "tpot_trend": tpot_trend,
        }
        
    def _detect_slo_violation(self, performance: Dict) -> Optional[Dict]:
        """检测SLO违反"""
        violations = {}
        
        # 检查TTFT违反
        if performance.get("avg_ttft", 0) > self.slo_target.ttft_target_ms:
            violations["ttft"] = {
                "current": performance["avg_ttft"],
                "target": self.slo_target.ttft_target_ms,
                "violation_ratio": performance["avg_ttft"] / self.slo_target.ttft_target_ms,
            }
            
        # 检查TPOT违反
        if performance.get("avg_tpot", 0) > self.slo_target.tpot_target_ms:
            violations["tpot"] = {
                "current": performance["avg_tpot"],
                "target": self.slo_target.tpot_target_ms,
                "violation_ratio": performance["avg_tpot"] / self.slo_target.tpot_target_ms,
            }
            
        # 检查趋势恶化
        if performance.get("ttft_trend", 0) > 0.1:  # 10%增长趋势
            violations["ttft_trend"] = performance["ttft_trend"]
            
        if performance.get("tpot_trend", 0) > 0.1:  # 10%增长趋势
            violations["tpot_trend"] = performance["tpot_trend"]
            
        return violations if violations else None
        
    def _predict_optimal_allocation(self) -> Optional[ResourcePrediction]:
        """预测最优资源分配"""
        if len(self.performance_history) < 10:
            return None
            
        try:
            # 获取当前分配
            current_status = self.process_rotation_manager.get_status()
            current_allocation = current_status["current_sm_allocation"]
            current_prefill_pct = current_allocation["prefill_percentage"]
            current_decode_pct = current_allocation["decode_percentage"]
            
            # 分析不同配比的预期性能
            best_allocation = None
            best_score = float('inf')
            best_confidence = 0.0
            
            # 尝试不同的配比组合
            allocation_candidates = self._generate_allocation_candidates(
                current_prefill_pct, current_decode_pct
            )
            
            for prefill_pct, decode_pct in allocation_candidates:
                allocation = SMAllocation(
                    prefill_percentage=prefill_pct,
                    decode_percentage=decode_pct
                )
                
                # 预测这个配比下的性能
                predicted_performance = self._predict_performance_for_allocation(allocation)
                
                if predicted_performance:
                    # 计算SLO满足度得分
                    score = self._calculate_slo_score(predicted_performance)
                    confidence = predicted_performance.confidence
                    
                    if score < best_score and confidence >= self.min_confidence_threshold:
                        best_score = score
                        best_allocation = allocation
                        best_confidence = confidence
                        
            if best_allocation:
                return ResourcePrediction(
                    predicted_ttft=0.0,  # 这里应该填入实际预测值
                    predicted_tpot=0.0,  # 这里应该填入实际预测值
                    confidence=best_confidence,
                    recommended_allocation=best_allocation,
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Error predicting optimal allocation: {e}")
            return None
            
    def _generate_allocation_candidates(
        self, 
        current_prefill_pct: int, 
        current_decode_pct: int
    ) -> List[Tuple[int, int]]:
        """生成候选分配方案"""
        candidates = []
        
        # 基于当前性能问题生成候选方案
        recent_performance = self._analyze_current_performance()
        
        # 如果TTFT有问题，增加prefill资源
        if recent_performance.get("avg_ttft", 0) > self.slo_target.ttft_target_ms:
            for delta in [5, 10, 15, 20]:
                new_prefill = min(80, current_prefill_pct + delta)
                new_decode = max(20, 100 - new_prefill)
                candidates.append((new_prefill, new_decode))
                
        # 如果TPOT有问题，增加decode资源
        if recent_performance.get("avg_tpot", 0) > self.slo_target.tpot_target_ms:
            for delta in [5, 10, 15, 20]:
                new_decode = min(80, current_decode_pct + delta)
                new_prefill = max(20, 100 - new_decode)
                candidates.append((new_prefill, new_decode))
                
        # 添加一些平衡的候选方案
        balanced_candidates = [
            (30, 70), (40, 60), (50, 50), (60, 40), (70, 30)
        ]
        candidates.extend(balanced_candidates)
        
        # 去重并过滤
        unique_candidates = list(set(candidates))
        valid_candidates = [
            (p, d) for p, d in unique_candidates 
            if 20 <= p <= 80 and 20 <= d <= 80 and p + d <= 100
        ]
        
        return valid_candidates
        
    def _predict_performance_for_allocation(self, allocation: SMAllocation) -> Optional[ResourcePrediction]:
        """预测特定分配下的性能"""
        try:
            # 简化的性能预测模型
            # 实际实现中应该使用更复杂的机器学习模型
            
            recent_metrics = list(self.performance_history)[-20:]
            if not recent_metrics:
                return None
                
            # 基于历史数据和资源分配预测性能
            base_ttft = np.mean([m.ttft_p95 for m in recent_metrics])
            base_tpot = np.mean([m.tpot_p95 for m in recent_metrics])
            
            # 简单的线性模型：更多prefill资源 -> 更好的TTFT
            prefill_factor = allocation.prefill_percentage / 50.0  # 归一化到50%基准
            decode_factor = allocation.decode_percentage / 50.0
            
            predicted_ttft = base_ttft / prefill_factor
            predicted_tpot = base_tpot / decode_factor
            
            # 计算置信度（基于历史数据的稳定性）
            ttft_variance = np.var([m.ttft_p95 for m in recent_metrics])
            tpot_variance = np.var([m.tpot_p95 for m in recent_metrics])
            
            # 方差越小，置信度越高
            confidence = 1.0 / (1.0 + (ttft_variance + tpot_variance) / 1000.0)
            confidence = max(0.1, min(1.0, confidence))
            
            return ResourcePrediction(
                predicted_ttft=predicted_ttft,
                predicted_tpot=predicted_tpot,
                confidence=confidence,
                recommended_allocation=allocation,
            )
            
        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            return None
            
    def _calculate_slo_score(self, prediction: ResourcePrediction) -> float:
        """计算SLO满足度得分（越小越好）"""
        ttft_score = max(0, prediction.predicted_ttft - self.slo_target.ttft_target_ms)
        tpot_score = max(0, prediction.predicted_tpot - self.slo_target.tpot_target_ms)
        
        # 归一化得分
        ttft_normalized = ttft_score / self.slo_target.ttft_target_ms
        tpot_normalized = tpot_score / self.slo_target.tpot_target_ms
        
        # 加权总分
        total_score = ttft_normalized * 0.5 + tpot_normalized * 0.5
        
        return total_score
        
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势（正值表示上升趋势）"""
        if len(values) < 2:
            return 0.0
            
        # 简单的线性回归斜率
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        # 归一化为百分比变化
        if np.mean(y) > 0:
            trend = slope / np.mean(y)
        else:
            trend = 0.0
            
        return trend
        
    def _update_slo_violations(self, metrics: WorkloadMetrics):
        """更新SLO违反统计"""
        self.slo_violations["total_measurements"] += 1
        
        if metrics.ttft_p95 > self.slo_target.ttft_target_ms:
            self.slo_violations["ttft_violations"] += 1
            
        if metrics.tpot_p95 > self.slo_target.tpot_target_ms:
            self.slo_violations["tpot_violations"] += 1
            
    def get_slo_compliance_rate(self) -> Dict:
        """获取SLO合规率"""
        total = self.slo_violations["total_measurements"]
        if total == 0:
            return {"ttft_compliance": 1.0, "tpot_compliance": 1.0, "overall_compliance": 1.0}
            
        ttft_compliance = 1.0 - (self.slo_violations["ttft_violations"] / total)
        tpot_compliance = 1.0 - (self.slo_violations["tpot_violations"] / total)
        overall_compliance = min(ttft_compliance, tpot_compliance)
        
        return {
            "ttft_compliance": ttft_compliance,
            "tpot_compliance": tpot_compliance,
            "overall_compliance": overall_compliance,
            "total_measurements": total,
        }
        
    def get_status(self) -> Dict:
        """获取管理器状态"""
        compliance = self.get_slo_compliance_rate()
        
        return {
            "running": self.running,
            "window_size_seconds": self.window_size_seconds,
            "metrics_count": len(self.metrics_window),
            "history_count": len(self.performance_history),
            "last_adjustment_time": self.last_adjustment_time,
            "slo_target": {
                "ttft_target_ms": self.slo_target.ttft_target_ms,
                "tpot_target_ms": self.slo_target.tpot_target_ms,
            },
            "slo_compliance": compliance,
            "slo_violations": self.slo_violations.copy(),
        }