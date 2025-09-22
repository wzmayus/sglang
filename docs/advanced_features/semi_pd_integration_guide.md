# Semi-PD功能集成指南

本文档详细介绍如何在SGLang中使用已整合的Semi-PD三个核心功能：

1. **常驻进程+进程轮转机制**
2. **SLO-aware动态资源调整算法**
3. **Unified Memory Manager**

## 快速开始

### 启用所有Semi-PD功能

```bash
# 启用Semi-PD协调器（包含所有功能）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator
```

### 选择性启用功能

```bash
# 只启用统一内存管理器
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-unified-memory

# 只启用SLO感知算法
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-slo-aware \
    --slo-ttft-target 80.0 \
    --slo-tpot-target 40.0
```

## 详细配置

### 1. 基本配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-semi-pd` | False | 启用Semi-PD（向后兼容） |
| `--enable-semi-pd-coordinator` | False | 启用Semi-PD协调器（推荐） |
| `--enable-unified-memory` | False | 启用统一内存管理器 |
| `--enable-slo-aware` | False | 启用SLO感知算法 |

### 2. Unified Memory Manager配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--unified-memory-blocks` | 1000 | 总内存块数 |
| `--unified-memory-block-size` | 4096 | 内存块大小（字节） |
| `--unified-memory-page-size` | 16 | 内存页大小 |

### 3. SLO配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--slo-ttft-target` | 100.0 | TTFT目标延迟（毫秒） |
| `--slo-tpot-target` | 50.0 | TPOT目标延迟（毫秒） |
| `--slo-window-size` | 30.0 | SLO监控窗口大小（秒） |
| `--slo-monitoring-interval` | 10.0 | SLO监控间隔（秒） |

### 4. SM分配配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--initial-prefill-sm` | 70 | 初始Prefill SM百分比 |
| `--initial-decode-sm` | 30 | 初始Decode SM百分比 |

## 使用场景

### 场景1：高吞吐量服务

适用于需要处理大量并发请求的场景。

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator \
    --unified-memory-blocks 2000 \
    --initial-prefill-sm 80 \
    --initial-decode-sm 20 \
    --slo-ttft-target 150.0 \
    --slo-tpot-target 60.0
```

**特点**：
- 更多内存块支持更多并发请求
- 偏向prefill的SM分配适合处理新请求
- 相对宽松的SLO目标确保稳定性

### 场景2：低延迟服务

适用于对响应时间要求严格的场景。

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator \
    --initial-prefill-sm 60 \
    --initial-decode-sm 40 \
    --slo-ttft-target 50.0 \
    --slo-tpot-target 25.0 \
    --slo-monitoring-interval 5.0
```

**特点**：
- 平衡的SM分配
- 严格的SLO目标
- 更频繁的监控确保快速响应

### 场景3：长文本生成

适用于需要生成长文本的场景。

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator \
    --initial-prefill-sm 40 \
    --initial-decode-sm 60 \
    --unified-memory-blocks 1500 \
    --unified-memory-block-size 8192 \
    --slo-tpot-target 30.0
```

**特点**：
- 偏向decode的SM分配
- 更大的内存块支持长序列
- 重点优化TPOT性能

## 编程接口

### 使用Semi-PD协调器

```python
from sglang.semi_pd.semi_pd_coordinator import create_semi_pd_coordinator
from sglang.semi_pd.slo_algorithm import SLOConstraints
from sglang.semi_pd.process_rotation_manager import SMAllocation
from sglang.srt.server_args import ServerArgs, SemiPDPortArgs

# 创建服务器参数
server_args = ServerArgs()
server_args.model_path = "meta-llama/Llama-3.1-8B-Instruct"
server_args.enable_semi_pd_coordinator = True
server_args.enable_unified_memory = True
server_args.enable_slo_aware = True

# 创建端口参数
port_args = SemiPDPortArgs()
port_args.host = "127.0.0.1"
port_args.port = 30000

# 创建协调器
coordinator = create_semi_pd_coordinator(
    server_args=server_args,
    port_args=port_args,
)

# 启动协调器
with coordinator:
    # 获取状态
    status = coordinator.get_status()
    print(f"协调器状态: {status['is_running']}")
    
    # 请求SM重新分配
    new_allocation = SMAllocation(prefill_percentage=80, decode_percentage=20)
    success = coordinator.request_sm_reallocation(new_allocation)
    
    # 获取内存使用情况
    memory_usage = coordinator.get_unified_memory_usage()
    if memory_usage:
        print(f"内存利用率: {memory_usage['utilization_ratio']:.1%}")
```

### 直接使用Unified Memory Manager

```python
from sglang.semi_pd.unified_memory_manager import UnifiedMemoryManager
from sglang.semi_pd.utils import InstanceRole
import torch

# 创建统一内存管理器
memory_manager = UnifiedMemoryManager(
    total_blocks=1000,
    block_size=4096,
    page_size=16,
    device=torch.device("cuda"),
)

# 分配KV cache块
block_ids = memory_manager.allocate_kv_cache_blocks(
    request_id="req_1",
    layer_id=0,
    num_blocks=10,
    requester_role=InstanceRole.PREFILL,
)

if block_ids:
    print(f"分配了 {len(block_ids)} 个块")
    
    # 获取块表索引
    block_indices = memory_manager.get_block_table_index("req_1", 0)
    print(f"块表索引: {block_indices}")
    
    # 访问KV cache
    kv_tensors = memory_manager.access_kv_cache(
        request_id="req_1",
        layer_id=0,
        accessor_role=InstanceRole.PREFILL,
    )
    
    # 释放块
    memory_manager.deallocate_kv_cache_blocks("req_1")
```

### 使用SLO-aware算法

```python
from sglang.semi_pd.slo_algorithm import SLOConstraints, SLOAwareResourceController
from sglang.semi_pd.metrics_collector import SemiPDMetricsCollector

# 创建SLO约束
slo_constraints = SLOConstraints(
    ttft_target_ms=100.0,
    tpot_target_ms=50.0,
)

# 创建metrics收集器
metrics_collector = SemiPDMetricsCollector(window_size_seconds=30.0)

# 创建SLO控制器
slo_controller = SLOAwareResourceController(
    slo_constraints=slo_constraints,
    metrics_collector=metrics_collector,
    process_rotation_manager=rotation_manager,  # 需要先创建
)

# 启动监控
slo_controller.start_monitoring()

# 记录请求指标
metrics_collector.record_request_arrival("req_1", input_length=1024)
metrics_collector.record_first_token("req_1")
metrics_collector.record_request_completion("req_1", output_length=256)

# 获取实时指标
real_time_metrics = metrics_collector.get_real_time_metrics()
print(f"TTFT P95: {real_time_metrics['ttft_p95_ms']:.1f}ms")
print(f"TPOT P95: {real_time_metrics['tpot_p95_ms']:.1f}ms")

# 停止监控
slo_controller.stop_monitoring()
```

## 监控和调试

### 1. 状态监控

```python
# 获取协调器综合状态
status = coordinator.get_status()

print("=== 协调器状态 ===")
print(f"运行状态: {status['is_running']}")
print(f"组件状态: {status['components']}")

# 获取各组件详细状态
if 'process_rotation_status' in status:
    rotation_status = status['process_rotation_status']
    print(f"当前SM分配: {rotation_status['current_sm_allocation']}")

if 'unified_memory_status' in status:
    memory_status = status['unified_memory_status']
    print(f"内存使用: {memory_status['memory_usage']}")

if 'slo_controller_status' in status:
    slo_status = status['slo_controller_status']
    print(f"SLO统计: {slo_status['statistics']}")
```

### 2. 性能指标

```python
# 获取实时性能指标
metrics = coordinator.metrics_aggregator.get_aggregated_metrics()

print("=== 性能指标 ===")
print(f"TTFT P95: {metrics.get('ttft_p95_ms', 0):.1f}ms")
print(f"TPOT P95: {metrics.get('tpot_p95_ms', 0):.1f}ms")
print(f"请求吞吐量: {metrics.get('request_throughput', 0):.1f} req/s")
print(f"Prefill队列长度: {metrics.get('prefill_queue_length', 0)}")
print(f"Decode队列长度: {metrics.get('decode_queue_length', 0)}")
```

### 3. 内存使用监控

```python
# 获取内存使用详情
memory_usage = coordinator.get_unified_memory_usage()

if memory_usage:
    print("=== 内存使用 ===")
    print(f"总内存: {memory_usage['total_memory_bytes']/1024/1024:.1f} MB")
    print(f"已用内存: {memory_usage['allocated_memory_bytes']/1024/1024:.1f} MB")
    print(f"利用率: {memory_usage['utilization_ratio']:.1%}")
    
    # 检查内存压力
    if memory_usage['utilization_ratio'] > 0.9:
        print("⚠️  内存使用率过高，建议增加内存块数量")
    elif memory_usage['utilization_ratio'] < 0.3:
        print("ℹ️  内存使用率较低，可以考虑减少内存块数量")
```

## 故障排除

### 常见问题

#### 1. 协调器启动失败

**症状**：启动时出现"Failed to initialize Semi-PD Coordinator"错误

**解决方案**：
```bash
# 检查参数配置
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator \
    --log-level DEBUG  # 启用详细日志
```

**常见原因**：
- SM分配百分比超过100%
- 内存块配置不合理
- 端口冲突

#### 2. SLO频繁违反

**症状**：日志中频繁出现"SLO VIOLATION DETECTED"

**解决方案**：
```bash
# 调整SLO目标
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator \
    --slo-ttft-target 150.0 \  # 放宽TTFT目标
    --slo-tpot-target 75.0 \   # 放宽TPOT目标
    --slo-monitoring-interval 15.0  # 减少监控频率
```

#### 3. 内存分配失败

**症状**：出现"Failed to allocate KV cache blocks"错误

**解决方案**：
```bash
# 增加内存块数量
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator \
    --unified-memory-blocks 2000 \  # 增加块数量
    --unified-memory-block-size 2048  # 减小块大小
```

#### 4. 进程轮转失败

**症状**：SM重新分配请求失败

**解决方案**：
- 检查MPS配置是否正确
- 确认GPU支持SM分区
- 验证进程间通信正常

### 调试工具

#### 1. 启用详细日志

```bash
export SGLANG_LOG_LEVEL=DEBUG
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-semi-pd-coordinator
```

#### 2. 运行集成测试

```bash
# 运行集成测试验证功能
python test/semi_pd/test_integration.py
```

#### 3. 运行演示程序

```bash
# 运行完整演示
python examples/semi_pd/integrated_example.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

## 性能优化建议

### 1. 内存配置优化

```bash
# 根据模型大小调整内存配置
# 对于7B模型
--unified-memory-blocks 1000 \
--unified-memory-block-size 4096

# 对于13B模型
--unified-memory-blocks 1500 \
--unified-memory-block-size 6144

# 对于70B模型
--unified-memory-blocks 3000 \
--unified-memory-block-size 8192
```

### 2. SLO目标设置

```bash
# 根据硬件性能设置合理的SLO目标
# 高端GPU (A100, H100)
--slo-ttft-target 50.0 \
--slo-tpot-target 25.0

# 中端GPU (RTX 4090, V100)
--slo-ttft-target 100.0 \
--slo-tpot-target 50.0

# 入门GPU (RTX 3080, T4)
--slo-ttft-target 200.0 \
--slo-tpot-target 100.0
```

### 3. SM分配策略

```bash
# 根据工作负载特点调整SM分配
# 短文本生成（偏向prefill）
--initial-prefill-sm 80 \
--initial-decode-sm 20

# 长文本生成（偏向decode）
--initial-prefill-sm 40 \
--initial-decode-sm 60

# 混合工作负载（平衡分配）
--initial-prefill-sm 60 \
--initial-decode-sm 40
```

## 总结

Semi-PD的三个核心功能已成功整合到SGLang中，提供了：

1. **统一的配置接口**：通过命令行参数轻松配置所有功能
2. **协调器模式**：自动管理和协调各个组件
3. **灵活的部署选项**：可以选择性启用需要的功能
4. **完整的监控支持**：实时监控性能和资源使用情况
5. **故障恢复机制**：自动处理异常情况并恢复服务

通过合理配置和使用这些功能，可以显著提升SGLang的性能和资源利用效率。