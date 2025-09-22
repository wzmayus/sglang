#!/usr/bin/env python3
"""
Semi-PD集成功能完整示例

展示如何在SGLang中使用三个核心功能：
1. 常驻进程+进程轮转机制
2. SLO-aware动态资源调整算法
3. Unified Memory Manager

使用方法:
python examples/semi_pd/integrated_example.py --model-path <MODEL_PATH>
"""

import argparse
import asyncio
import logging
import time
from typing import List

import torch

from sglang.semi_pd.semi_pd_coordinator import create_semi_pd_coordinator
from sglang.semi_pd.process_rotation_manager import SMAllocation
from sglang.semi_pd.slo_algorithm import SLOConstraints
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import ServerArgs, SemiPDPortArgs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemiPDIntegratedDemo:
    """Semi-PD集成功能演示"""
    
    def __init__(self, args):
        self.args = args
        self.coordinator = None
        
        # 配置服务器参数
        self.server_args = self._create_server_args()
        self.port_args = self._create_port_args()
        
        logger.info("Semi-PD集成演示初始化完成")
        
    def _create_server_args(self) -> ServerArgs:
        """创建服务器参数"""
        server_args = ServerArgs()
        
        # 基本配置
        server_args.model_path = self.args.model_path
        server_args.host = self.args.host
        server_args.port = self.args.port
        server_args.tp_size = self.args.tp_size
        
        # 启用Semi-PD功能
        server_args.enable_semi_pd = True
        server_args.enable_semi_pd_coordinator = True
        server_args.enable_unified_memory = True
        server_args.enable_slo_aware = True
        
        # Unified Memory Manager配置
        server_args.unified_memory_blocks = self.args.memory_blocks
        server_args.unified_memory_block_size = self.args.memory_block_size
        server_args.unified_memory_page_size = self.args.memory_page_size
        
        # SLO配置
        server_args.slo_ttft_target = self.args.ttft_target
        server_args.slo_tpot_target = self.args.tpot_target
        server_args.slo_window_size = self.args.slo_window_size
        server_args.slo_monitoring_interval = self.args.slo_monitoring_interval
        
        # SM分配配置
        server_args.initial_prefill_sm = self.args.initial_prefill_sm
        server_args.initial_decode_sm = self.args.initial_decode_sm
        
        return server_args
        
    def _create_port_args(self) -> SemiPDPortArgs:
        """创建端口参数"""
        port_args = SemiPDPortArgs()
        port_args.host = self.args.host
        port_args.port = self.args.port
        return port_args
        
    def demo_coordinator_creation(self):
        """演示协调器创建"""
        logger.info("=== 演示协调器创建 ===")
        
        # 创建协调器
        self.coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        logger.info("✅ Semi-PD协调器创建成功")
        
        # 显示配置信息
        status = self.coordinator.get_status()
        config = status["configuration"]
        
        logger.info("协调器配置:")
        logger.info(f"  - SLO感知算法: {'启用' if config['slo_enabled'] else '禁用'}")
        logger.info(f"  - 统一内存管理: {'启用' if config['unified_memory_enabled'] else '禁用'}")
        logger.info(f"  - 初始SM分配: P{config['initial_sm_allocation']['prefill_percentage']}% D{config['initial_sm_allocation']['decode_percentage']}%")
        
    def demo_unified_memory_manager(self):
        """演示统一内存管理器"""
        logger.info("\n=== 演示统一内存管理器 ===")
        
        if not self.coordinator.unified_memory_manager:
            logger.warning("统一内存管理器未启用")
            return
            
        # 模拟多个请求的内存分配
        requests = [
            {"request_id": "req_1", "layers": [0, 1, 2], "blocks_per_layer": 4},
            {"request_id": "req_2", "layers": [0, 1, 2, 3], "blocks_per_layer": 6},
            {"request_id": "req_3", "layers": [0, 1], "blocks_per_layer": 8},
        ]
        
        allocated_requests = []
        
        for req in requests:
            logger.info(f"为请求 {req['request_id']} 分配内存...")
            
            request_success = True
            for layer_id in req["layers"]:
                # 随机选择角色
                role = InstanceRole.PREFILL if layer_id % 2 == 0 else InstanceRole.DECODE
                
                block_ids = self.coordinator.allocate_kv_cache_blocks(
                    request_id=req["request_id"],
                    layer_id=layer_id,
                    num_blocks=req["blocks_per_layer"],
                    requester_role=role,
                )
                
                if block_ids:
                    logger.info(f"  ✅ 层 {layer_id}: 分配 {len(block_ids)} 个块 (角色: {role.name})")
                else:
                    logger.error(f"  ❌ 层 {layer_id}: 分配失败")
                    request_success = False
                    break
                    
            if request_success:
                allocated_requests.append(req)
                
        # 显示内存使用情况
        usage = self.coordinator.get_unified_memory_usage()
        if usage:
            logger.info("内存使用情况:")
            logger.info(f"  - 总块数: {usage['total_blocks']}")
            logger.info(f"  - 已分配: {usage['allocated_blocks']} ({usage['utilization_ratio']:.1%})")
            logger.info(f"  - 空闲: {usage['free_blocks']} ({usage['free_ratio']:.1%})")
            
        # 释放部分内存
        for req in allocated_requests[:2]:
            success = self.coordinator.deallocate_kv_cache_blocks(req["request_id"])
            if success:
                logger.info(f"✅ 释放请求 {req['request_id']} 的内存")
                
    def demo_slo_aware_algorithm(self):
        """演示SLO感知算法"""
        logger.info("\n=== 演示SLO感知算法 ===")
        
        if not self.coordinator.slo_constraints:
            logger.warning("SLO感知算法未启用")
            return
            
        # 显示SLO配置
        slo = self.coordinator.slo_constraints
        logger.info("SLO配置:")
        logger.info(f"  - TTFT目标: {slo.ttft_target_ms} ms")
        logger.info(f"  - TPOT目标: {slo.tpot_target_ms} ms")
        logger.info(f"  - 违反阈值: {slo.ttft_violation_threshold:.1%}")
        
        # 模拟指标更新
        scenarios = [
            {
                "name": "正常负载",
                "prefill_metrics": {
                    "ttft_p95_ms": 75.0,
                    "prefill_queue_length": 5,
                    "prefill_utilization": 0.7,
                    "input_throughput": 1000.0,
                },
                "decode_metrics": {
                    "tpot_p95_ms": 35.0,
                    "decode_queue_length": 8,
                    "decode_utilization": 0.6,
                    "output_throughput": 2000.0,
                },
            },
            {
                "name": "高负载（可能违反SLO）",
                "prefill_metrics": {
                    "ttft_p95_ms": 120.0,  # 超过目标
                    "prefill_queue_length": 15,
                    "prefill_utilization": 0.9,
                    "input_throughput": 800.0,
                },
                "decode_metrics": {
                    "tpot_p95_ms": 60.0,  # 超过目标
                    "decode_queue_length": 20,
                    "decode_utilization": 0.95,
                    "output_throughput": 1500.0,
                },
            },
        ]
        
        for scenario in scenarios:
            logger.info(f"\n模拟场景: {scenario['name']}")
            
            # 更新指标
            self.coordinator.update_metrics(
                scenario["prefill_metrics"],
                scenario["decode_metrics"]
            )
            
            # 检查是否需要调整
            prefill_metrics = scenario["prefill_metrics"]
            decode_metrics = scenario["decode_metrics"]
            
            ttft_violation = prefill_metrics["ttft_p95_ms"] > slo.ttft_target_ms
            tpot_violation = decode_metrics["tpot_p95_ms"] > slo.tpot_target_ms
            
            if ttft_violation or tpot_violation:
                logger.warning("🚨 检测到SLO违反!")
                if ttft_violation:
                    logger.warning(f"  - TTFT违反: {prefill_metrics['ttft_p95_ms']:.1f}ms > {slo.ttft_target_ms}ms")
                if tpot_violation:
                    logger.warning(f"  - TPOT违反: {decode_metrics['tpot_p95_ms']:.1f}ms > {slo.tpot_target_ms}ms")
                    
                # 模拟资源调整
                logger.info("触发资源调整...")
                if ttft_violation:
                    # TTFT违反，增加prefill资源
                    new_allocation = SMAllocation(prefill_percentage=80, decode_percentage=20)
                else:
                    # TPOT违反，增加decode资源
                    new_allocation = SMAllocation(prefill_percentage=50, decode_percentage=50)
                    
                logger.info(f"建议新的SM分配: P{new_allocation.prefill_percentage}% D{new_allocation.decode_percentage}%")
            else:
                logger.info("✅ SLO约束满足")
                
            time.sleep(1)  # 模拟时间间隔
            
    def demo_process_rotation(self):
        """演示进程轮转"""
        logger.info("\n=== 演示进程轮转机制 ===")
        
        # 显示当前SM分配
        current_allocation = self.coordinator.get_current_sm_allocation()
        if current_allocation:
            logger.info(f"当前SM分配: P{current_allocation.prefill_percentage}% D{current_allocation.decode_percentage}%")
        else:
            logger.info("当前SM分配: 未知（进程轮换管理器未启动）")
            
        # 模拟SM重新分配请求
        test_allocations = [
            SMAllocation(prefill_percentage=80, decode_percentage=20),
            SMAllocation(prefill_percentage=60, decode_percentage=40),
            SMAllocation(prefill_percentage=50, decode_percentage=50),
        ]
        
        for allocation in test_allocations:
            logger.info(f"请求SM重新分配: P{allocation.prefill_percentage}% D{allocation.decode_percentage}%")
            
            # 注意：在实际环境中，这会触发真正的进程轮转
            # 这里只是演示API调用
            success = self.coordinator.request_sm_reallocation(allocation)
            
            if success:
                logger.info("✅ SM重新分配请求已提交")
            else:
                logger.warning("❌ SM重新分配请求失败")
                
            time.sleep(2)  # 模拟调整时间
            
    def demo_comprehensive_status(self):
        """演示综合状态监控"""
        logger.info("\n=== 演示综合状态监控 ===")
        
        status = self.coordinator.get_status()
        
        logger.info("协调器状态:")
        logger.info(f"  - 运行状态: {'运行中' if status['is_running'] else '已停止'}")
        
        components = status["components"]
        logger.info("组件状态:")
        logger.info(f"  - 进程轮换管理器: {'✓' if components['process_rotation_manager'] else '✗'}")
        logger.info(f"  - 统一内存管理器: {'✓' if components['unified_memory_manager'] else '✗'}")
        logger.info(f"  - SLO控制器: {'✓' if components['slo_controller'] else '✗'}")
        logger.info(f"  - 指标聚合器: {'✓' if components['metrics_aggregator'] else '✗'}")
        
        # 显示详细状态（如果可用）
        if "unified_memory_status" in status:
            memory_status = status["unified_memory_status"]
            logger.info("统一内存管理器详细状态:")
            logger.info(f"  - 配置: {memory_status['configuration']}")
            
        if "aggregated_metrics" in status:
            metrics = status["aggregated_metrics"]
            logger.info("聚合指标:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  - {key}: {value}")
                    
    def run_demo(self):
        """运行完整演示"""
        logger.info("开始Semi-PD集成功能演示")
        logger.info("=" * 50)
        
        try:
            # 1. 创建协调器
            self.demo_coordinator_creation()
            
            # 2. 演示统一内存管理器
            self.demo_unified_memory_manager()
            
            # 3. 演示SLO感知算法
            self.demo_slo_aware_algorithm()
            
            # 4. 演示进程轮转
            self.demo_process_rotation()
            
            # 5. 演示综合状态监控
            self.demo_comprehensive_status()
            
            logger.info("\n" + "=" * 50)
            logger.info("✅ Semi-PD集成功能演示完成！")
            logger.info("\n三个核心功能已成功整合:")
            logger.info("1. 常驻进程+进程轮转机制")
            logger.info("2. SLO-aware动态资源调整算法")
            logger.info("3. Unified Memory Manager")
            
        except Exception as e:
            logger.error(f"演示过程中出错: {e}")
            raise
        finally:
            # 清理资源
            if self.coordinator:
                self.coordinator.stop()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Semi-PD集成功能演示")
    
    # 基本参数
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="模型路径"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=30000, help="端口")
    parser.add_argument("--tp-size", type=int, default=1, help="张量并行大小")
    
    # Unified Memory Manager参数
    parser.add_argument("--memory-blocks", type=int, default=200, help="内存块数量")
    parser.add_argument("--memory-block-size", type=int, default=1024, help="内存块大小")
    parser.add_argument("--memory-page-size", type=int, default=16, help="内存页大小")
    
    # SLO参数
    parser.add_argument("--ttft-target", type=float, default=80.0, help="TTFT目标延迟（毫秒）")
    parser.add_argument("--tpot-target", type=float, default=40.0, help="TPOT目标延迟（毫秒）")
    parser.add_argument("--slo-window-size", type=float, default=30.0, help="SLO窗口大小（秒）")
    parser.add_argument("--slo-monitoring-interval", type=float, default=5.0, help="SLO监控间隔（秒）")
    
    # SM分配参数
    parser.add_argument("--initial-prefill-sm", type=int, default=60, help="初始Prefill SM百分比")
    parser.add_argument("--initial-decode-sm", type=int, default=40, help="初始Decode SM百分比")
    
    # 日志级别
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 验证参数
    if args.initial_prefill_sm + args.initial_decode_sm > 100:
        print("错误: Prefill SM + Decode SM 不能超过 100%")
        return 1
        
    # 运行演示
    demo = SemiPDIntegratedDemo(args)
    
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        logger.info("演示被用户中断")
    except Exception as e:
        logger.error(f"演示失败: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())