#!/usr/bin/env python3
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
Semi-PD功能使用示例

本示例展示如何使用Semi-PD的新功能：
1. 进程轮换机制
2. 统一存储管理器
3. SLO感知的动态资源调整

使用方法:
python examples/semi_pd/semi_pd_example.py
"""

import argparse
import logging
import time
from typing import Dict, List

import torch

from sglang.semi_pd.semi_pd_coordinator import create_semi_pd_coordinator
from sglang.semi_pd.slo_aware_resource_manager import SLOTarget, WorkloadMetrics
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import PortArgs, ServerArgs, SemiPDPortArgs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemiPDDemo:
    """Semi-PD功能演示类"""
    
    def __init__(self, args):
        self.args = args
        self.coordinator = None
        self.running = False
        
    def setup_coordinator(self):
        """设置协调器"""
        logger.info("Setting up Semi-PD coordinator...")
        
        # 创建服务器参数
        server_args = ServerArgs()
        server_args.model_path = self.args.model_path
        server_args.enable_prefix_caching = True
        server_args.tp_size = 1
        server_args.dp_size = 1
        
        # 创建端口参数
        port_args = SemiPDPortArgs()
        port_args.host = "127.0.0.1"
        port_args.port = 30000
        
        # 创建协调器
        self.coordinator = create_semi_pd_coordinator(
            server_args=server_args,
            port_args=port_args,
            prefill_sm_percentage=self.args.initial_prefill_sm,
            decode_sm_percentage=self.args.initial_decode_sm,
            ttft_target_ms=self.args.ttft_target,
            tpot_target_ms=self.args.tpot_target,
            total_kv_blocks=self.args.total_kv_blocks,
            kv_block_size=4096,
        )
        
        logger.info("Semi-PD coordinator setup completed")
        
    def start_coordinator(self):
        """启动协调器"""
        logger.info("Starting Semi-PD coordinator...")
        
        try:
            self.coordinator.start()
            self.running = True
            logger.info("Semi-PD coordinator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start coordinator: {e}")
            raise
            
    def stop_coordinator(self):
        """停止协调器"""
        if self.coordinator and self.running:
            logger.info("Stopping Semi-PD coordinator...")
            self.coordinator.stop()
            self.running = False
            logger.info("Semi-PD coordinator stopped")
            
    def simulate_workload(self):
        """模拟工作负载"""
        logger.info("Starting workload simulation...")
        
        request_counter = 0
        
        try:
            for cycle in range(self.args.simulation_cycles):
                logger.info(f"Simulation cycle {cycle + 1}/{self.args.simulation_cycles}")
                
                # 模拟请求分配
                self._simulate_request_allocation(cycle, request_counter)
                
                # 模拟性能指标
                self._simulate_performance_metrics(cycle)
                
                # 显示状态
                if cycle % 5 == 0:
                    self._display_status()
                    
                # 等待下一个周期
                time.sleep(self.args.cycle_interval)
                
                request_counter += 10  # 每个周期增加10个请求
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            
    def _simulate_request_allocation(self, cycle: int, base_request_id: int):
        """模拟请求分配"""
        # 模拟不同类型的请求
        num_prefill_requests = 3 + (cycle % 5)
        num_decode_requests = 5 + (cycle % 3)
        
        # 分配prefill请求
        for i in range(num_prefill_requests):
            request_id = f"prefill_req_{base_request_id + i}"
            blocks_needed = 5 + (i % 3)
            
            handles = self.coordinator.allocate_kv_blocks(
                request_id=request_id,
                num_blocks=blocks_needed,
                requester_role=InstanceRole.PREFILL,
            )
            
            if handles:
                logger.debug(f"Allocated {blocks_needed} blocks for {request_id}")
            else:
                logger.warning(f"Failed to allocate blocks for {request_id}")
                
        # 分配decode请求
        for i in range(num_decode_requests):
            request_id = f"decode_req_{base_request_id + num_prefill_requests + i}"
            blocks_needed = 8 + (i % 4)
            
            handles = self.coordinator.allocate_kv_blocks(
                request_id=request_id,
                num_blocks=blocks_needed,
                requester_role=InstanceRole.DECODE,
            )
            
            if handles:
                logger.debug(f"Allocated {blocks_needed} blocks for {request_id}")
            else:
                logger.warning(f"Failed to allocate blocks for {request_id}")
                
        # 随机释放一些旧请求
        if cycle > 2:
            old_cycle = cycle - 2
            old_base = base_request_id - 20
            
            for i in range(2):
                old_request_id = f"prefill_req_{old_base + i}"
                success = self.coordinator.deallocate_kv_blocks(old_request_id)
                if success:
                    logger.debug(f"Deallocated blocks for {old_request_id}")
                    
    def _simulate_performance_metrics(self, cycle: int):
        """模拟性能指标"""
        # 模拟不同的负载模式
        if cycle < 10:
            # 轻负载
            ttft = 60.0 + (cycle * 2)
            tpot = 30.0 + (cycle * 1)
            prefill_util = 0.3 + (cycle * 0.02)
            decode_util = 0.4 + (cycle * 0.02)
        elif cycle < 20:
            # 中等负载
            ttft = 80.0 + ((cycle - 10) * 3)
            tpot = 40.0 + ((cycle - 10) * 2)
            prefill_util = 0.5 + ((cycle - 10) * 0.03)
            decode_util = 0.6 + ((cycle - 10) * 0.03)
        else:
            # 高负载（可能违反SLO）
            ttft = 110.0 + ((cycle - 20) * 5)
            tpot = 60.0 + ((cycle - 20) * 3)
            prefill_util = 0.8 + ((cycle - 20) * 0.02)
            decode_util = 0.9 + ((cycle - 20) * 0.01)
            
        # 创建指标对象
        metrics = WorkloadMetrics(
            timestamp=time.time(),
            prefill_queue_length=max(0, int(prefill_util * 20)),
            decode_queue_length=max(0, int(decode_util * 30)),
            prefill_throughput=1000.0 * (1.0 - prefill_util),
            decode_throughput=2000.0 * (1.0 - decode_util),
            ttft_p95=ttft,
            tpot_p95=tpot,
            prefill_utilization=min(1.0, prefill_util),
            decode_utilization=min(1.0, decode_util),
        )
        
        # 报告指标
        self.coordinator.report_workload_metrics(metrics)
        
        logger.info(
            f"Metrics - TTFT: {ttft:.1f}ms, TPOT: {tpot:.1f}ms, "
            f"P_util: {prefill_util:.2f}, D_util: {decode_util:.2f}"
        )
        
    def _display_status(self):
        """显示状态信息"""
        logger.info("=== Semi-PD Status ===")
        
        try:
            # 获取综合状态
            status = self.coordinator.get_comprehensive_status()
            
            # 显示协调器状态
            coord_stats = status.get("coordinator", {}).get("stats", {})
            logger.info(f"Total requests: {coord_stats.get('total_requests', 0)}")
            logger.info(f"Successful allocations: {coord_stats.get('successful_allocations', 0)}")
            logger.info(f"Failed allocations: {coord_stats.get('failed_allocations', 0)}")
            logger.info(f"Resource adjustments: {coord_stats.get('resource_adjustments', 0)}")
            
            # 显示内存使用
            memory_usage = status.get("memory_usage", {})
            logger.info(f"Memory utilization: {memory_usage.get('utilization', 0):.2%}")
            logger.info(f"Free blocks: {memory_usage.get('free_blocks', 0)}")
            
            # 显示SLO合规性
            slo_compliance = status.get("slo_compliance", {})
            logger.info(f"TTFT compliance: {slo_compliance.get('ttft_compliance', 0):.2%}")
            logger.info(f"TPOT compliance: {slo_compliance.get('tpot_compliance', 0):.2%}")
            
            # 显示进程状态
            process_status = status.get("process_rotation", {})
            logger.info(f"Switch requested: {process_status.get('switch_requested', False)}")
            
            # 健康检查
            health = self.coordinator.health_check()
            logger.info(f"System healthy: {health.get('healthy', False)}")
            if health.get('issues'):
                logger.warning(f"Issues: {health['issues']}")
                
        except Exception as e:
            logger.error(f"Error displaying status: {e}")
            
        logger.info("=====================")
        
    def run_demo(self):
        """运行演示"""
        logger.info("Starting Semi-PD demonstration")
        
        try:
            # 设置和启动协调器
            self.setup_coordinator()
            self.start_coordinator()
            
            # 运行模拟
            self.simulate_workload()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # 清理
            self.stop_coordinator()
            
        logger.info("Semi-PD demonstration completed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Semi-PD功能演示")
    
    # 模型参数
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="模型路径"
    )
    
    # SM分配参数
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
    
    # 资源配置
    parser.add_argument(
        "--total-kv-blocks",
        type=int,
        default=1000,
        help="KV cache总块数"
    )
    
    # 模拟参数
    parser.add_argument(
        "--simulation-cycles",
        type=int,
        default=30,
        help="模拟周期数"
    )
    parser.add_argument(
        "--cycle-interval",
        type=float,
        default=2.0,
        help="周期间隔（秒）"
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
    demo = SemiPDDemo(args)
    
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