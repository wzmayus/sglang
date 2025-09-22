#!/usr/bin/env python3
"""
SLO感知动态资源调整示例

展示如何：
1. 获取SGLang运行过程中的真实metrics
2. 应用论文第5节的SLO-aware adjusting algorithm
3. 实现动态资源调整

使用方法:
python examples/semi_pd/slo_aware_example.py --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import logging
import time
from typing import Dict, List

import numpy as np

from sglang.semi_pd.metrics_collector import SemiPDMetricsCollector, SystemMetrics
from sglang.semi_pd.metrics_integration import MetricsAggregator
from sglang.semi_pd.process_rotation_manager import ProcessRotationManager, SMAllocation
from sglang.semi_pd.slo_algorithm import SLOConstraints, SLOAwareResourceController
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import ServerArgs, SemiPDPortArgs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SLOAwareDemo:
    """SLO感知动态资源调整演示"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化组件
        self.setup_components()
        
    def setup_components(self):
        """设置组件"""
        logger.info("Setting up SLO-aware demo components...")
        
        # 服务器参数
        self.server_args = ServerArgs()
        self.server_args.model_path = self.args.model_path
        self.server_args.enable_semi_pd = True
        
        # 端口参数
        self.port_args = SemiPDPortArgs()
        self.port_args.host = "127.0.0.1"
        self.port_args.port = 30000
        
        # SLO约束
        self.slo_constraints = SLOConstraints(
            ttft_target_ms=self.args.ttft_target,
            tpot_target_ms=self.args.tpot_target,
            ttft_percentile=95.0,
            tpot_percentile=95.0,
        )
        
        # 初始资源分配
        self.initial_allocation = SMAllocation(
            prefill_percentage=self.args.initial_prefill_sm,
            decode_percentage=self.args.initial_decode_sm,
        )
        
        # 进程轮换管理器
        self.process_rotation_manager = ProcessRotationManager(
            server_args=self.server_args,
            port_args=self.port_args,
            initial_sm_allocation=self.initial_allocation,
            gpu_id=0,
            tp_rank=0,
        )
        
        # Metrics收集器
        self.prefill_metrics_collector = SemiPDMetricsCollector(window_size_seconds=30.0)
        self.decode_metrics_collector = SemiPDMetricsCollector(window_size_seconds=30.0)
        self.metrics_aggregator = MetricsAggregator()
        
        # SLO感知资源控制器
        self.slo_controller = SLOAwareResourceController(
            slo_constraints=self.slo_constraints,
            metrics_collector=self.prefill_metrics_collector,  # 使用prefill作为主要收集器
            process_rotation_manager=self.process_rotation_manager,
            monitoring_interval=self.args.monitoring_interval,
        )
        
        logger.info("Components setup completed")
        
    def simulate_real_workload(self):
        """模拟真实工作负载"""
        logger.info("Starting real workload simulation...")
        
        request_id_counter = 0
        
        try:
            for cycle in range(self.args.simulation_cycles):
                logger.info(f"Simulation cycle {cycle + 1}/{self.args.simulation_cycles}")
                
                # 模拟请求到达和处理
                self._simulate_request_processing(cycle, request_id_counter)
                
                # 模拟系统指标
                self._simulate_system_metrics(cycle)
                
                # 更新聚合指标
                self._update_aggregated_metrics()
                
                # 显示当前状态
                if cycle % 5 == 0:
                    self._display_metrics_and_slo_status()
                    
                request_id_counter += 20  # 每个周期20个请求
                
                # 等待下一个周期
                time.sleep(self.args.cycle_interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            
    def _simulate_request_processing(self, cycle: int, base_request_id: int):
        """模拟请求处理过程"""
        # 模拟不同负载模式
        if cycle < 10:
            # 轻负载阶段
            num_requests = 5 + (cycle % 3)
            avg_input_length = 512
            avg_output_length = 128
        elif cycle < 20:
            # 中等负载阶段
            num_requests = 10 + (cycle % 5)
            avg_input_length = 1024
            avg_output_length = 256
        else:
            # 高负载阶段（可能违反SLO）
            num_requests = 15 + (cycle % 7)
            avg_input_length = 2048
            avg_output_length = 512
            
        current_time = time.time()
        
        # 模拟请求处理流程
        for i in range(num_requests):
            request_id = f"req_{base_request_id + i}"
            
            # 随机化输入输出长度
            input_length = max(100, int(np.random.normal(avg_input_length, avg_input_length * 0.2)))
            output_length = max(50, int(np.random.normal(avg_output_length, avg_output_length * 0.3)))
            
            # 模拟请求到达
            self.prefill_metrics_collector.record_request_arrival(request_id, input_length)
            
            # 模拟prefill处理
            prefill_delay = self._calculate_prefill_delay(input_length, cycle)
            time.sleep(prefill_delay / 1000.0)  # 转换为秒
            
            self.prefill_metrics_collector.record_prefill_start(request_id)
            self.prefill_metrics_collector.record_first_token(request_id)
            self.prefill_metrics_collector.record_prefill_end(request_id)
            
            # 模拟decode处理
            decode_delay = self._calculate_decode_delay(output_length, cycle)
            time.sleep(decode_delay / 1000.0)  # 转换为秒
            
            self.decode_metrics_collector.record_decode_start(request_id)
            
            # 模拟缓存命中
            cached_tokens = int(input_length * np.random.uniform(0.1, 0.4))
            
            # 完成请求
            self.prefill_metrics_collector.record_request_completion(
                request_id, output_length, cached_tokens, success=True
            )
            self.decode_metrics_collector.record_request_completion(
                request_id, output_length, cached_tokens, success=True
            )
            
    def _calculate_prefill_delay(self, input_length: int, cycle: int) -> float:
        """计算prefill延迟（毫秒）"""
        # 基础延迟 + 长度相关延迟 + 负载相关延迟
        base_delay = 50.0
        length_delay = input_length * 0.05  # 每个token 0.05ms
        load_delay = cycle * 2.0  # 随着周期增加负载
        
        # 添加一些随机性
        noise = np.random.normal(0, 10.0)
        
        return max(20.0, base_delay + length_delay + load_delay + noise)
        
    def _calculate_decode_delay(self, output_length: int, cycle: int) -> float:
        """计算decode延迟（毫秒）"""
        # 每个token的延迟
        per_token_delay = 15.0 + cycle * 0.5  # 随着周期增加延迟
        
        # 添加一些随机性
        noise = np.random.normal(0, 2.0)
        
        return max(10.0, output_length * (per_token_delay + noise))
        
    def _simulate_system_metrics(self, cycle: int):
        """模拟系统指标"""
        current_time = time.time()
        
        # 模拟队列长度（随负载增加）
        base_queue_length = min(50, cycle * 2)
        prefill_queue = max(0, int(np.random.normal(base_queue_length, base_queue_length * 0.3)))
        decode_queue = max(0, int(np.random.normal(base_queue_length * 1.5, base_queue_length * 0.4)))
        
        # 模拟利用率
        base_utilization = min(0.9, cycle * 0.03)
        prefill_util = max(0.1, min(1.0, np.random.normal(base_utilization, 0.1)))
        decode_util = max(0.1, min(1.0, np.random.normal(base_utilization * 1.2, 0.1)))
        
        # 模拟吞吐量
        prefill_throughput = max(100, 2000 - cycle * 30)
        decode_throughput = max(200, 3000 - cycle * 40)
        
        # 模拟缓存命中率
        cache_hit_rate = max(0.1, min(0.8, 0.6 - cycle * 0.01))
        
        # 记录prefill系统指标
        prefill_system_metrics = SystemMetrics(
            timestamp=current_time,
            prefill_queue_length=prefill_queue,
            prefill_running_requests=min(20, prefill_queue + 5),
            prefill_utilization=prefill_util,
            prefill_throughput=prefill_throughput,
            cache_hit_rate=cache_hit_rate,
        )
        
        # 记录decode系统指标
        decode_system_metrics = SystemMetrics(
            timestamp=current_time,
            decode_queue_length=decode_queue,
            decode_running_requests=min(30, decode_queue + 8),
            decode_utilization=decode_util,
            decode_throughput=decode_throughput,
            cache_hit_rate=cache_hit_rate,
        )
        
        self.prefill_metrics_collector.record_system_metrics(prefill_system_metrics)
        self.decode_metrics_collector.record_system_metrics(decode_system_metrics)
        
    def _update_aggregated_metrics(self):
        """更新聚合指标"""
        prefill_metrics = self.prefill_metrics_collector.get_real_time_metrics()
        decode_metrics = self.decode_metrics_collector.get_real_time_metrics()
        
        self.metrics_aggregator.update_prefill_metrics(prefill_metrics)
        self.metrics_aggregator.update_decode_metrics(decode_metrics)
        
    def _display_metrics_and_slo_status(self):
        """显示指标和SLO状态"""
        logger.info("=== Current Metrics & SLO Status ===")
        
        try:
            # 获取聚合指标
            aggregated_metrics = self.metrics_aggregator.get_aggregated_metrics()
            
            if aggregated_metrics:
                logger.info(f"TTFT P95: {aggregated_metrics.get('ttft_p95_ms', 0):.1f}ms "
                           f"(Target: {self.slo_constraints.ttft_target_ms}ms)")
                logger.info(f"TPOT P95: {aggregated_metrics.get('tpot_p95_ms', 0):.1f}ms "
                           f"(Target: {self.slo_constraints.tpot_target_ms}ms)")
                
                # 检查SLO违反
                ttft_violation = aggregated_metrics.get('ttft_p95_ms', 0) > self.slo_constraints.ttft_target_ms
                tpot_violation = aggregated_metrics.get('tpot_p95_ms', 0) > self.slo_constraints.tpot_target_ms
                
                if ttft_violation or tpot_violation:
                    logger.warning("🚨 SLO VIOLATION DETECTED!")
                    if ttft_violation:
                        logger.warning(f"  - TTFT violation: {aggregated_metrics.get('ttft_p95_ms', 0):.1f}ms > {self.slo_constraints.ttft_target_ms}ms")
                    if tpot_violation:
                        logger.warning(f"  - TPOT violation: {aggregated_metrics.get('tpot_p95_ms', 0):.1f}ms > {self.slo_constraints.tpot_target_ms}ms")
                else:
                    logger.info("✅ SLO constraints satisfied")
                    
                logger.info(f"Queue lengths - Prefill: {aggregated_metrics.get('prefill_queue_length', 0)}, "
                           f"Decode: {aggregated_metrics.get('decode_queue_length', 0)}")
                logger.info(f"Utilization - Prefill: {aggregated_metrics.get('prefill_utilization', 0):.2%}, "
                           f"Decode: {aggregated_metrics.get('decode_utilization', 0):.2%}")
                logger.info(f"Throughput - Total: {aggregated_metrics.get('total_throughput', 0):.1f} tokens/s")
                
            # 获取当前资源分配
            status = self.process_rotation_manager.get_status()
            current_allocation = status["current_sm_allocation"]
            logger.info(f"Current SM allocation - Prefill: {current_allocation['prefill_percentage']}%, "
                       f"Decode: {current_allocation['decode_percentage']}%")
            
            # 显示SLO控制器状态
            controller_status = self.slo_controller.get_controller_status()
            stats = controller_status["statistics"]
            logger.info(f"SLO Controller - Adjustments: {stats['total_adjustments']}, "
                       f"Success rate: {stats['success_rate']:.1%}, "
                       f"Violations detected: {stats['slo_violations_detected']}")
                       
        except Exception as e:
            logger.error(f"Error displaying metrics: {e}")
            
        logger.info("=====================================")
        
    def run_demo(self):
        """运行演示"""
        logger.info("Starting SLO-aware dynamic resource adjustment demo")
        
        try:
            # 启动进程轮换管理器（模拟模式）
            logger.info("Starting process rotation manager...")
            # 注意：在实际环境中需要调用 self.process_rotation_manager.start()
            
            # 启动SLO监控
            logger.info("Starting SLO monitoring...")
            self.slo_controller.start_monitoring()
            
            # 运行工作负载模拟
            self.simulate_real_workload()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # 清理
            logger.info("Stopping SLO monitoring...")
            self.slo_controller.stop_monitoring()
            
            logger.info("Stopping process rotation manager...")
            # 注意：在实际环境中需要调用 self.process_rotation_manager.stop()
            
        logger.info("SLO-aware demo completed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SLO感知动态资源调整演示")
    
    # 模型参数
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="模型路径"
    )
    
    # SLO目标
    parser.add_argument(
        "--ttft-target",
        type=float,
        default=100.0,
        help="TTFT目标延迟（毫秒）"
    )
    parser.add_argument(
        "--tpot-target",
        type=float,
        default=50.0,
        help="TPOT目标延迟（毫秒）"
    )
    
    # 初始资源分配
    parser.add_argument(
        "--initial-prefill-sm",
        type=int,
        default=70,
        help="初始Prefill SM百分比"
    )
    parser.add_argument(
        "--initial-decode-sm",
        type=int,
        default=30,
        help="初始Decode SM百分比"
    )
    
    # 模拟参数
    parser.add_argument(
        "--simulation-cycles",
        type=int,
        default=50,
        help="模拟周期数"
    )
    parser.add_argument(
        "--cycle-interval",
        type=float,
        default=2.0,
        help="周期间隔（秒）"
    )
    parser.add_argument(
        "--monitoring-interval",
        type=float,
        default=5.0,
        help="SLO监控间隔（秒）"
    )
    
    # 日志级别
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 运行演示
    demo = SLOAwareDemo(args)
    
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())