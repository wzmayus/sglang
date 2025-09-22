# Semi-PD Metrics收集与SLO感知算法

本文档详细介绍如何获取SGLang运行过程中的真实metrics，并根据论文第5节实现SLO感知的动态资源调整算法。

## 概述

根据论文 <mcreference link="https://arxiv.org/html/2504.19867v1#S5" index="0">0</mcreference> 第5节，Semi-PD实现了SLO感知的动态资源调整算法，主要目标是：

1. **监控关键性能指标**：TTFT (Time to First Token)、TPOT (Time per Output Token)
2. **检测SLO违反**：当性能指标超过预设阈值时触发调整
3. **动态资源调整**：在满足SLO约束的前提下最大化吞吐量
4. **预测性能影响**：评估不同资源配置对性能的影响

## 1. SGLang运行时Metrics收集

### 1.1 现有Metrics系统

SGLang已有完善的metrics收集系统：

```python
# SGLang内置的metrics收集器
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats

# 主要收集的指标
@dataclass
class SchedulerStats:
    num_running_reqs: int = 0          # 运行中的请求数
    num_used_tokens: int = 0           # 使用的token数
    token_usage: float = 0.0           # token使用率
    gen_throughput: float = 0.0        # 生成吞吐量
    num_queue_reqs: int = 0            # 队列中的请求数
    avg_request_queue_latency: float = 0.0  # 平均队列延迟
    cache_hit_rate: float = 0.0        # 缓存命中率
```

### 1.2 Semi-PD专用Metrics收集器

我们扩展了现有系统，创建了Semi-PD专用的metrics收集器：

```python
from sglang.semi_pd.metrics_collector import SemiPDMetricsCollector, RequestMetrics, SystemMetrics

# 创建metrics收集器
metrics_collector = SemiPDMetricsCollector(
    window_size_seconds=30.0,  # 30秒滑动窗口
    max_history_size=1000,     # 最大历史记录数
)

# 记录请求生命周期
metrics_collector.record_request_arrival("req_123", input_length=1024)
metrics_collector.record_prefill_start("req_123")
metrics_collector.record_first_token("req_123")  # TTFT计算点
metrics_collector.record_prefill_end("req_123")
metrics_collector.record_decode_start("req_123")
metrics_collector.record_request_completion("req_123", output_length=256, success=True)
```

### 1.3 关键指标定义

#### TTFT (Time to First Token)
```python
@property
def ttft_ms(self) -> Optional[float]:
    """Time to First Token (毫秒)"""
    if self.first_token_time and self.arrival_time:
        return (self.first_token_time - self.arrival_time) * 1000
    return None
```

#### TPOT (Time per Output Token)
```python
@property
def tpot_ms(self) -> Optional[float]:
    """Time per Output Token (毫秒)"""
    if (self.completion_time and self.first_token_time and 
        self.output_length > 1):
        decode_time = self.completion_time - self.first_token_time
        return (decode_time / (self.output_length - 1)) * 1000
    return None
```

### 1.4 与现有Scheduler集成

通过Mixin模式集成到现有scheduler：

```python
from sglang.semi_pd.metrics_integration import integrate_semi_pd_metrics_with_scheduler

# 集成到scheduler
integrate_semi_pd_metrics_with_scheduler(scheduler, InstanceRole.PREFILL)

# 在scheduler中使用
class SemiPDPrefillScheduler(Scheduler):
    def handle_generate_request(self, recv_req):
        # 记录请求到达
        self.semi_pd_on_request_arrival(req)
        
        # 原有处理逻辑...
        
    def run_batch(self, batch):
        # 记录prefill开始
        self.semi_pd_on_batch_prefill_start(batch)
        
        # 执行prefill
        result = super().run_batch(batch)
        
        # 记录prefill结束和第一个token
        self.semi_pd_on_batch_prefill_end(batch)
        for req in batch.reqs:
            self.semi_pd_on_first_token_generated(req)
            
        return result
```

## 2. SLO感知算法实现

### 2.1 SLO约束定义

```python
from sglang.semi_pd.slo_algorithm import SLOConstraints

slo_constraints = SLOConstraints(
    ttft_target_ms=100.0,      # TTFT目标：100ms
    tpot_target_ms=50.0,       # TPOT目标：50ms
    ttft_percentile=95.0,      # 使用P95百分位数
    tpot_percentile=95.0,      # 使用P95百分位数
    ttft_violation_threshold=1.1,  # 超过目标10%算违反
    tpot_violation_threshold=1.1,  # 超过目标10%算违反
)
```

### 2.2 SLO违反检测

```python
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
```

### 2.3 资源分配候选方案生成

根据论文算法，生成多个资源分配候选方案：

```python
def _generate_allocation_candidates(self, current_allocation: SMAllocation) -> List[ResourceAllocationCandidate]:
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
                
    return candidates
```

### 2.4 性能预测模型

```python
def _predict_performance(self, candidate: ResourceAllocationCandidate, current_metrics: AggregatedMetrics) -> None:
    """预测候选方案的性能"""
    # 基准性能
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
    throughput_factor = 0.8 + 0.4 * balance_factor
    candidate.predicted_throughput = base_throughput * throughput_factor
```

### 2.5 最优方案选择

```python
def _select_best_candidate(self, candidates: List[ResourceAllocationCandidate]) -> Optional[ResourceAllocationCandidate]:
    """选择最佳候选方案"""
    if not candidates:
        return None
        
    # 评估每个候选方案
    for candidate in candidates:
        # 检查是否满足SLO约束
        ttft_satisfies_slo = candidate.predicted_ttft <= self.slo_constraints.ttft_target_ms
        tpot_satisfies_slo = candidate.predicted_tpot <= self.slo_constraints.tpot_target_ms
        
        if ttft_satisfies_slo and tpot_satisfies_slo:
            # 在满足SLO的前提下最大化吞吐量
            candidate.overall_score = -candidate.predicted_throughput
        else:
            candidate.overall_score = float('inf')
    
    # 按综合得分排序（得分越小越好）
    candidates.sort(key=lambda c: c.overall_score)
    
    return candidates[0] if candidates[0].overall_score != float('inf') else None
```

## 3. 完整的SLO感知资源控制器

### 3.1 控制器初始化

```python
from sglang.semi_pd.slo_algorithm import SLOAwareResourceController

# 创建SLO感知资源控制器
slo_controller = SLOAwareResourceController(
    slo_constraints=slo_constraints,
    metrics_collector=metrics_collector,
    process_rotation_manager=rotation_manager,
    monitoring_interval=10.0,  # 10秒监控间隔
)

# 启动监控
slo_controller.start_monitoring()
```

### 3.2 监控循环

```python
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
                # 计算最优分配
                optimal_allocation = self.slo_algorithm.compute_optimal_allocation(
                    current_metrics, current_allocation
                )
                
                if optimal_allocation:
                    # 请求资源调整
                    success = self.process_rotation_manager.request_sm_reallocation(optimal_allocation)
                    
            time.sleep(self.monitoring_interval)
            
        except Exception as e:
            logger.error(f"Error in SLO monitoring loop: {e}")
```

## 4. 跨进程Metrics聚合

在Semi-PD架构中，prefill和decode运行在不同进程中，需要聚合指标：

```python
from sglang.semi_pd.metrics_integration import MetricsAggregator

# 创建聚合器
aggregator = MetricsAggregator()

# 更新各进程的指标
aggregator.update_prefill_metrics(prefill_metrics)
aggregator.update_decode_metrics(decode_metrics)

# 获取聚合后的指标
aggregated_metrics = aggregator.get_aggregated_metrics()
```

## 5. 实际使用示例

### 5.1 基本使用

```python
#!/usr/bin/env python3
"""基本的SLO感知资源调整示例"""

from sglang.semi_pd.metrics_collector import SemiPDMetricsCollector
from sglang.semi_pd.slo_algorithm import SLOConstraints, SLOAwareResourceController
from sglang.semi_pd.process_rotation_manager import ProcessRotationManager, SMAllocation

def main():
    # 1. 设置SLO约束
    slo_constraints = SLOConstraints(
        ttft_target_ms=100.0,  # TTFT目标100ms
        tpot_target_ms=50.0,   # TPOT目标50ms
    )
    
    # 2. 创建metrics收集器
    metrics_collector = SemiPDMetricsCollector(window_size_seconds=30.0)
    
    # 3. 创建进程轮换管理器
    initial_allocation = SMAllocation(prefill_percentage=70, decode_percentage=30)
    rotation_manager = ProcessRotationManager(
        server_args=server_args,
        port_args=port_args,
        initial_sm_allocation=initial_allocation,
    )
    
    # 4. 创建SLO控制器
    slo_controller = SLOAwareResourceController(
        slo_constraints=slo_constraints,
        metrics_collector=metrics_collector,
        process_rotation_manager=rotation_manager,
    )
    
    # 5. 启动系统
    rotation_manager.start()
    slo_controller.start_monitoring()
    
    try:
        # 运行服务...
        time.sleep(3600)  # 运行1小时
    finally:
        # 清理
        slo_controller.stop_monitoring()
        rotation_manager.stop()

if __name__ == "__main__":
    main()
```

### 5.2 与现有Scheduler集成

```python
# 在Semi-PD scheduler中集成metrics收集
class SemiPDPrefillScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 集成Semi-PD metrics
        integrate_semi_pd_metrics_with_scheduler(self, InstanceRole.PREFILL)
        
    def handle_generate_request(self, recv_req):
        # 记录请求到达
        req = self.create_request(recv_req)
        self.semi_pd_on_request_arrival(req)
        
        # 原有处理逻辑
        return super().handle_generate_request(recv_req)
        
    def forward_batch_generation(self, batch):
        # 记录批次处理开始
        self.semi_pd_on_batch_prefill_start(batch)
        
        # 执行forward
        result = super().forward_batch_generation(batch)
        
        # 记录批次处理结束
        self.semi_pd_on_batch_prefill_end(batch)
        
        # 记录第一个token生成
        for req in batch.reqs:
            if req.is_first_token():
                self.semi_pd_on_first_token_generated(req)
                
        return result
```

## 6. 监控和调试

### 6.1 实时监控

```python
# 获取实时指标
real_time_metrics = metrics_collector.get_real_time_metrics()
print(f"TTFT P95: {real_time_metrics['ttft_p95_ms']:.1f}ms")
print(f"TPOT P95: {real_time_metrics['tpot_p95_ms']:.1f}ms")
print(f"Request throughput: {real_time_metrics['request_throughput']:.1f} req/s")

# 获取SLO控制器状态
controller_status = slo_controller.get_controller_status()
print(f"Total adjustments: {controller_status['statistics']['total_adjustments']}")
print(f"Success rate: {controller_status['statistics']['success_rate']:.1%}")
```

### 6.2 性能分析

```python
# 获取聚合指标用于分析
aggregated_metrics = metrics_collector.get_current_window_metrics()

print(f"Window: {aggregated_metrics.window_start_time} - {aggregated_metrics.window_end_time}")
print(f"TTFT stats: mean={aggregated_metrics.ttft_mean:.1f}, p95={aggregated_metrics.ttft_p95:.1f}")
print(f"TPOT stats: mean={aggregated_metrics.tpot_mean:.1f}, p95={aggregated_metrics.tpot_p95:.1f}")
print(f"Throughput: {aggregated_metrics.request_throughput:.1f} req/s")
```

## 7. 运行演示

我们提供了完整的演示程序：

```bash
# 运行SLO感知算法演示
python examples/semi_pd/slo_aware_example.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --ttft-target 100.0 \
    --tpot-target 50.0 \
    --initial-prefill-sm 70 \
    --initial-decode-sm 30 \
    --simulation-cycles 50
```

演示程序会：
1. 模拟真实的工作负载
2. 收集TTFT、TPOT等关键指标
3. 检测SLO违反情况
4. 自动调整SM资源分配
5. 显示调整效果

## 8. 性能优化建议

### 8.1 Metrics收集优化

1. **合理设置窗口大小**：太小会导致指标不稳定，太大会延迟响应
2. **控制历史记录大小**：避免内存占用过多
3. **异步收集**：避免影响主要服务性能

### 8.2 SLO算法优化

1. **调整冷却期**：避免频繁调整导致系统不稳定
2. **改进预测模型**：使用机器学习模型提高预测准确性
3. **多目标优化**：平衡延迟、吞吐量和资源利用率

### 8.3 系统集成优化

1. **进程间通信优化**：使用高效的IPC机制传递指标
2. **资源调整优化**：实现渐进式调整而非突变
3. **故障恢复**：处理metrics收集失败的情况

## 总结

本文档详细介绍了如何在SGLang中实现Semi-PD的metrics收集和SLO感知算法。通过这套系统，可以：

1. ✅ **实时收集关键性能指标**：TTFT、TPOT、吞吐量等
2. ✅ **自动检测SLO违反**：基于P95百分位数的阈值检测
3. ✅ **智能资源调整**：在满足SLO约束下最大化吞吐量
4. ✅ **预测性能影响**：评估资源调整对性能的影响
5. ✅ **跨进程指标聚合**：处理Semi-PD分离架构的复杂性

这套实现完全符合论文第5节的算法设计，为Semi-PD提供了完整的SLO感知动态资源调整能力。