# Semi-PD 新功能

本文档介绍了Semi-PD架构的两个重要新功能：进程轮换机制和SLO感知的动态资源调整机制。

## 概述

Semi-PD（Semi Prefill-Decode）是一种创新的推理引擎架构，采用"计算分离存储融合"的设计理念。在原有基础上，我们新增了以下功能：

1. **进程轮换机制**：支持动态调整SM配比而不阻塞当前推理
2. **统一存储管理器**：确保KV cache分配的原子性和读写一致性  
3. **SLO感知的动态资源调整**：基于实时负载自动优化资源分配

## 功能详述

### 1. 进程轮换机制

#### 背景
在MPS（Multi-Process Service）中，无法对已有进程的SM用量进行动态调整，因此调整SM配比需要重新启动进程。为了避免阻塞当前推理，我们实现了两组进程轮换机制。

#### 实现原理
- **推理进程组**：当前正在处理请求的进程
- **休眠进程组**：预先启动的备用进程，使用新的SM配比

当需要调整资源时：
1. 休眠进程用新配比重新启动
2. 推理进程感知到切换信号后准备交接
3. 在推理进程当前step结束后进行角色切换
4. 完成SM配比的更新

#### 核心组件

**ProcessRotationManager**
```python
from sglang.semi_pd.process_rotation_manager import ProcessRotationManager, SMAllocation

# 创建SM分配配置
allocation = SMAllocation(prefill_percentage=70, decode_percentage=30)

# 初始化进程轮换管理器
rotation_manager = ProcessRotationManager(
    server_args=server_args,
    port_args=port_args,
    initial_sm_allocation=allocation,
)

# 启动管理器
rotation_manager.start()

# 请求资源重新分配
new_allocation = SMAllocation(prefill_percentage=60, decode_percentage=40)
success = rotation_manager.request_sm_reallocation(new_allocation)
```

### 2. 统一存储管理器

#### 背景
在Semi-PD架构中，P和D是异步的，但KV cache的存储是共享的。我们需要确保cache block的分配具有原子性，并由decode实例作为KV cache的主要管理者。

#### 核心特性
- **原子性**：块分配操作的原子性保证
- **一致性**：P和D实例之间的读写一致性
- **并发安全**：多进程访问的线程安全
- **内存效率**：避免内存碎片和泄漏

#### 使用示例

**UnifiedStorageManager**
```python
from sglang.semi_pd.unified_storage_manager import UnifiedStorageManager
from sglang.semi_pd.utils import InstanceRole

# 初始化存储管理器
storage_manager = UnifiedStorageManager(
    total_blocks=10000,
    block_size=4096,
    device=torch.device("cuda"),
    enable_prefix_caching=True,
)

# 分配KV cache块
block_ids = storage_manager.allocate_blocks(
    request_id="req_123",
    num_blocks=10,
    requester_role=InstanceRole.PREFILL,
)

# 获取块句柄用于跨进程共享
handles = storage_manager.get_block_handles("req_123")

# 释放块
storage_manager.deallocate_blocks("req_123")

# 块共享（prefix caching）
storage_manager.share_blocks("source_req", "target_req")
```

### 3. SLO感知的动态资源调整机制

#### 背景
该机制能够根据实时负载情况，动态调整资源划分比例(x, y)，以更好地满足延迟约束和系统吞吐的双重目标。

#### 工作原理
1. **周期性负载感知**：设置窗口大小，周期性感知系统负载
2. **SLO违反检测**：监控TTFT和TPOT是否违反目标
3. **资源配比预测**：通过预测(x,y)配比下的TTFT和TPOT找到最优调整点
4. **动态调整触发**：当检测到SLO违反时触发资源重新分配

#### 核心组件

**SLOAwareResourceManager**
```python
from sglang.semi_pd.slo_aware_resource_manager import (
    SLOAwareResourceManager, SLOTarget, WorkloadMetrics
)

# 定义SLO目标
slo_target = SLOTarget(
    ttft_target_ms=100.0,  # TTFT目标：100ms
    tpot_target_ms=50.0,   # TPOT目标：50ms
)

# 初始化SLO管理器
slo_manager = SLOAwareResourceManager(
    slo_target=slo_target,
    process_rotation_manager=rotation_manager,
    window_size_seconds=30.0,
    adjustment_cooldown_seconds=60.0,
)

# 启动管理器
slo_manager.start()

# 报告工作负载指标
metrics = WorkloadMetrics(
    timestamp=time.time(),
    prefill_queue_length=5,
    decode_queue_length=10,
    prefill_throughput=1000.0,
    decode_throughput=2000.0,
    ttft_p95=80.0,
    tpot_p95=40.0,
    prefill_utilization=0.7,
    decode_utilization=0.8,
)
slo_manager.report_metrics(metrics)
```

## 统一协调器

为了简化使用，我们提供了`SemiPDCoordinator`来统一管理所有组件：

```python
from sglang.semi_pd.semi_pd_coordinator import create_semi_pd_coordinator

# 创建协调器
coordinator = create_semi_pd_coordinator(
    server_args=server_args,
    port_args=port_args,
    prefill_sm_percentage=70,
    decode_sm_percentage=30,
    ttft_target_ms=100.0,
    tpot_target_ms=50.0,
)

# 启动协调器
coordinator.start()

# 分配KV块
handles = coordinator.allocate_kv_blocks("req_id", 10, InstanceRole.PREFILL)

# 报告指标
coordinator.report_workload_metrics(metrics)

# 获取状态
status = coordinator.get_comprehensive_status()

# 健康检查
health = coordinator.health_check()
```

## 配置参数

### 环境变量
- `SEMI_PD_PREFILL_SM_PERCENTILE`: Prefill进程SM百分比（默认80）
- `SEMI_PD_DECODE_SM_PERCENTILE`: Decode进程SM百分比（默认100）

### 关键参数
- `window_size_seconds`: SLO监控窗口大小（默认30秒）
- `adjustment_cooldown_seconds`: 资源调整冷却时间（默认60秒）
- `min_confidence_threshold`: 预测置信度阈值（默认0.7）

## 性能优势

根据我们的测试，相比于开源的SOTA实现：
- **Goodput提升**: 1.55-1.72x
- **端到端延迟提升**: 1.27-2.58x
- **资源利用率**: 显著提高
- **SLO合规性**: 更好的延迟保证

## 使用示例

完整的使用示例请参考：
```bash
python examples/semi_pd/semi_pd_example.py --help
```

## 测试

运行测试套件：
```bash
python -m pytest test/semi_pd/test_semi_pd_features.py -v
```

## 注意事项

1. **MPS要求**: 需要CUDA MPS支持
2. **内存管理**: 注意KV cache内存使用
3. **进程同步**: 确保进程间通信正常
4. **监控指标**: 及时报告工作负载指标以获得最佳性能

## 故障排除

### 常见问题

1. **进程切换失败**
   - 检查MPS配置
   - 确认进程间通信正常
   - 查看日志中的错误信息

2. **内存分配失败**
   - 检查可用内存
   - 调整块大小配置
   - 启用垃圾回收

3. **SLO违反**
   - 检查资源配比是否合理
   - 调整SLO目标值
   - 增加监控窗口大小

### 调试建议

1. 启用详细日志：`--log-level DEBUG`
2. 监控系统资源使用情况
3. 检查进程状态和健康检查结果
4. 分析SLO合规性报告

## 参考资料

- [Semi-PD技术报告](https://arxiv.org/abs/2504.19867)
- [SGLang项目主页](https://github.com/sgl-project/sglang)
- [CUDA MPS文档](https://docs.nvidia.com/deploy/mps/index.html)