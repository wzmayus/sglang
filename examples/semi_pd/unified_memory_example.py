#!/usr/bin/env python3
"""
Unified Memory Manager使用示例

展示论文4.4节描述的功能：
1. 模型权重的只读访问
2. KV cache的paged storage
3. WAR冲突的原子性分配
4. prefill和decode worker的异步访问

使用方法:
python examples/semi_pd/unified_memory_example.py
"""

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import torch
import numpy as np

from sglang.semi_pd.unified_memory_manager import UnifiedMemoryManager
from sglang.semi_pd.utils import InstanceRole

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedMemoryDemo:
    """Unified Memory Manager演示"""
    
    def __init__(self, args):
        self.args = args
        
        # 创建统一内存管理器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager = UnifiedMemoryManager(
            total_blocks=args.total_blocks,
            block_size=args.block_size,
            page_size=args.page_size,
            device=self.device,
            dtype=torch.float16,
            enable_prefix_caching=args.enable_prefix_caching,
        )
        
        logger.info(f"Unified Memory Manager initialized on {self.device}")
        logger.info(f"Total blocks: {args.total_blocks}, Block size: {args.block_size} bytes")
        
    def demo_model_weights_management(self):
        """演示模型权重管理（只读访问）"""
        logger.info("=== 演示模型权重管理 ===")
        
        # 模拟注册多个层的权重
        layer_configs = [
            ("embedding.weight", (50000, 4096)),
            ("layer1.attention.weight", (4096, 4096)),
            ("layer1.mlp.weight", (4096, 16384)),
            ("layer2.attention.weight", (4096, 4096)),
            ("layer2.mlp.weight", (4096, 16384)),
        ]
        
        for weight_name, shape in layer_configs:
            # 创建随机权重
            weight_tensor = torch.randn(shape, dtype=torch.float16, device=self.device)
            
            # 注册权重
            success = self.memory_manager.register_model_weights(weight_name, weight_tensor)
            if success:
                logger.info(f"✅ 注册权重: {weight_name}, 形状: {shape}")
            else:
                logger.error(f"❌ 注册权重失败: {weight_name}")
                
        # 模拟多个worker同时访问权重（只读）
        def weight_access_worker(worker_id, role):
            """权重访问工作线程"""
            access_count = 0
            for _ in range(10):
                for weight_name, _ in layer_configs:
                    weight = self.memory_manager.get_model_weights(weight_name)
                    if weight is not None:
                        access_count += 1
                        # 模拟使用权重进行计算
                        _ = weight.sum()
                time.sleep(0.01)  # 模拟计算时间
                
            logger.info(f"{role.name} worker {worker_id}: 访问权重 {access_count} 次")
            return access_count
            
        # 启动多个worker并发访问权重
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            # 3个prefill worker
            for i in range(3):
                future = executor.submit(weight_access_worker, i, InstanceRole.PREFILL)
                futures.append(future)
                
            # 3个decode worker
            for i in range(3):
                future = executor.submit(weight_access_worker, i, InstanceRole.DECODE)
                futures.append(future)
                
            # 等待完成
            total_accesses = sum(future.result() for future in as_completed(futures))
            
        logger.info(f"总权重访问次数: {total_accesses}")
        
        # 显示权重访问统计
        stats = self.memory_manager.get_statistics()
        logger.info("权重访问统计:")
        for weight_name, count in stats["weight_access_stats"].items():
            logger.info(f"  {weight_name}: {count} 次访问")
            
    def demo_kv_cache_allocation(self):
        """演示KV cache分配和paged storage"""
        logger.info("\n=== 演示KV cache分配和paged storage ===")
        
        # 模拟多个请求的KV cache分配
        requests = [
            {"request_id": "req_1", "layers": [0, 1, 2], "blocks_per_layer": 4},
            {"request_id": "req_2", "layers": [0, 1, 2, 3], "blocks_per_layer": 6},
            {"request_id": "req_3", "layers": [0, 1], "blocks_per_layer": 8},
        ]
        
        allocated_requests = []
        
        for req in requests:
            logger.info(f"分配请求 {req['request_id']} 的KV cache...")
            
            request_success = True
            for layer_id in req["layers"]:
                # 随机选择prefill或decode角色
                role = InstanceRole.PREFILL if np.random.random() > 0.5 else InstanceRole.DECODE
                
                block_ids = self.memory_manager.allocate_kv_cache_blocks(
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
        usage = self.memory_manager.get_memory_usage()
        logger.info(f"内存使用情况:")
        logger.info(f"  总块数: {usage['total_blocks']}")
        logger.info(f"  已分配: {usage['allocated_blocks']} ({usage['utilization_ratio']:.1%})")
        logger.info(f"  空闲: {usage['free_blocks']} ({usage['free_ratio']:.1%})")
        
        # 演示块表索引访问
        logger.info("\n演示块表索引访问:")
        for req in allocated_requests[:2]:  # 只演示前两个请求
            for layer_id in req["layers"][:2]:  # 只演示前两层
                block_indices = self.memory_manager.get_block_table_index(
                    req["request_id"], layer_id
                )
                if block_indices:
                    logger.info(f"  {req['request_id']} 层 {layer_id}: 块索引 {block_indices}")
                    
                    # 访问KV cache
                    kv_tensors = self.memory_manager.access_kv_cache(
                        request_id=req["request_id"],
                        layer_id=layer_id,
                        accessor_role=InstanceRole.DECODE,
                    )
                    if kv_tensors:
                        logger.info(f"    ✅ 成功访问 {len(kv_tensors)} 个KV tensor")
                        
        return allocated_requests
        
    def demo_war_conflict_resolution(self):
        """演示WAR冲突解决"""
        logger.info("\n=== 演示WAR冲突解决 ===")
        
        # 记录初始统计
        initial_stats = self.memory_manager.get_statistics()
        initial_failures = initial_stats["allocation_failures"]
        
        def concurrent_allocator(worker_id, role, num_requests=5):
            """并发分配器 - 模拟WAR冲突场景"""
            results = []
            
            for i in range(num_requests):
                request_id = f"{role.name.lower()}_worker_{worker_id}_req_{i}"
                
                # 每个请求分配多个层
                layer_allocations = []
                for layer_id in range(3):  # 3层
                    block_ids = self.memory_manager.allocate_kv_cache_blocks(
                        request_id=request_id,
                        layer_id=layer_id,
                        num_blocks=2,  # 每层2个块
                        requester_role=role,
                    )
                    
                    if block_ids:
                        layer_allocations.append((layer_id, block_ids))
                    else:
                        # 分配失败，清理已分配的层
                        self.memory_manager.deallocate_kv_cache_blocks(request_id)
                        break
                        
                success = len(layer_allocations) == 3
                results.append({
                    "request_id": request_id,
                    "success": success,
                    "allocated_layers": len(layer_allocations),
                })
                
                # 模拟一些处理时间
                time.sleep(0.001)
                
            return results
            
        logger.info("启动并发分配器模拟WAR冲突...")
        
        # 启动多个并发worker
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # 4个prefill worker
            for i in range(4):
                future = executor.submit(concurrent_allocator, i, InstanceRole.PREFILL, 3)
                futures.append(("prefill", future))
                
            # 4个decode worker
            for i in range(4):
                future = executor.submit(concurrent_allocator, i, InstanceRole.DECODE, 3)
                futures.append(("decode", future))
                
            # 收集结果
            all_results = []
            for worker_type, future in futures:
                try:
                    results = future.result()
                    all_results.extend(results)
                    
                    successful = sum(1 for r in results if r["success"])
                    logger.info(f"{worker_type} worker: {successful}/{len(results)} 请求成功")
                    
                except Exception as e:
                    logger.error(f"{worker_type} worker 出错: {e}")
                    
        # 分析结果
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r["success"])
        failed_requests = total_requests - successful_requests
        
        logger.info(f"并发分配结果:")
        logger.info(f"  总请求数: {total_requests}")
        logger.info(f"  成功: {successful_requests} ({successful_requests/total_requests:.1%})")
        logger.info(f"  失败: {failed_requests} ({failed_requests/total_requests:.1%})")
        
        # 检查WAR冲突
        conflicts = self.memory_manager.detect_war_conflicts()
        if conflicts:
            logger.warning(f"检测到 {len(conflicts)} 个WAR冲突:")
            for conflict in conflicts:
                logger.warning(f"  块 {conflict['block_id']}: {conflict['description']}")
        else:
            logger.info("✅ 未检测到WAR冲突 - 原子性分配机制工作正常")
            
        # 显示最终统计
        final_stats = self.memory_manager.get_statistics()
        new_failures = final_stats["allocation_failures"] - initial_failures
        
        logger.info(f"分配统计:")
        logger.info(f"  新增分配失败: {new_failures}")
        logger.info(f"  平均分配时间: {final_stats['avg_allocation_time']:.4f}s")
        logger.info(f"  峰值利用率: {final_stats['peak_utilization']:.1%}")
        
    def demo_memory_cleanup(self, allocated_requests):
        """演示内存清理"""
        logger.info("\n=== 演示内存清理 ===")
        
        # 显示清理前状态
        usage_before = self.memory_manager.get_memory_usage()
        logger.info(f"清理前: {usage_before['allocated_blocks']} 块已分配")
        
        # 释放一半的请求
        requests_to_free = allocated_requests[:len(allocated_requests)//2]
        
        for req in requests_to_free:
            success = self.memory_manager.deallocate_kv_cache_blocks(req["request_id"])
            if success:
                logger.info(f"✅ 释放请求 {req['request_id']} 的内存")
            else:
                logger.error(f"❌ 释放请求 {req['request_id']} 失败")
                
        # 显示清理后状态
        usage_after = self.memory_manager.get_memory_usage()
        logger.info(f"清理后: {usage_after['allocated_blocks']} 块已分配")
        
        freed_blocks = usage_before['allocated_blocks'] - usage_after['allocated_blocks']
        logger.info(f"释放了 {freed_blocks} 个块")
        
        # 强制垃圾回收
        logger.info("执行强制垃圾回收...")
        collected_blocks = self.memory_manager.force_garbage_collection()
        logger.info(f"垃圾回收释放了 {collected_blocks} 个块")
        
    def run_demo(self):
        """运行完整演示"""
        logger.info("开始Unified Memory Manager演示")
        logger.info("基于论文4.4节的实现")
        
        try:
            # 1. 模型权重管理
            self.demo_model_weights_management()
            
            # 2. KV cache分配和paged storage
            allocated_requests = self.demo_kv_cache_allocation()
            
            # 3. WAR冲突解决
            self.demo_war_conflict_resolution()
            
            # 4. 内存清理
            self.demo_memory_cleanup(allocated_requests)
            
            # 5. 最终状态报告
            logger.info("\n=== 最终状态报告 ===")
            status = self.memory_manager.get_status()
            
            logger.info("内存使用:")
            usage = status["memory_usage"]
            logger.info(f"  利用率: {usage['utilization_ratio']:.1%}")
            logger.info(f"  已分配内存: {usage['allocated_memory_bytes']/1024/1024:.1f} MB")
            logger.info(f"  空闲内存: {usage['free_memory_bytes']/1024/1024:.1f} MB")
            
            logger.info("统计信息:")
            stats = status["statistics"]
            logger.info(f"  总分配次数: {stats['total_allocations']}")
            logger.info(f"  总释放次数: {stats['total_deallocations']}")
            logger.info(f"  分配失败次数: {stats['allocation_failures']}")
            logger.info(f"  活跃请求数: {stats['active_requests']}")
            
            logger.info("✅ 演示完成！")
            
        except Exception as e:
            logger.error(f"演示过程中出错: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Unified Memory Manager演示")
    
    # 内存配置
    parser.add_argument(
        "--total-blocks",
        type=int,
        default=1000,
        help="总块数"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4096,
        help="块大小（字节）"
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="页大小"
    )
    
    # 功能开关
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="启用前缀缓存"
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
    demo = UnifiedMemoryDemo(args)
    
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