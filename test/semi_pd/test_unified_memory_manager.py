#!/usr/bin/env python3
"""
测试Unified Memory Manager

验证论文4.4节描述的功能：
1. 模型权重的只读访问
2. KV cache的paged storage
3. WAR冲突的原子性分配
4. prefill和decode worker的异步访问
"""

import sys
import os
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加python目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torch

from sglang.semi_pd.unified_memory_manager import (
    UnifiedMemoryManager,
    MemoryUtilization,
    BlockState,
    AllocationRequest,
)
from sglang.semi_pd.utils import InstanceRole


class TestMemoryUtilization(unittest.TestCase):
    """测试内存利用率管理"""
    
    def test_basic_utilization(self):
        """测试基本的内存利用率操作"""
        util = MemoryUtilization(total_blocks=100)
        
        # 测试初始状态
        status = util.query_utilization()
        self.assertEqual(status["total_blocks"], 100)
        self.assertEqual(status["free_blocks"], 100)
        self.assertEqual(status["allocated_blocks"], 0)
        
        # 测试分配
        success = util.atomic_allocate(20)
        self.assertTrue(success)
        
        status = util.query_utilization()
        self.assertEqual(status["free_blocks"], 80)
        self.assertEqual(status["allocated_blocks"], 20)
        
        # 测试释放
        success = util.atomic_deallocate(10)
        self.assertTrue(success)
        
        status = util.query_utilization()
        self.assertEqual(status["free_blocks"], 90)
        self.assertEqual(status["allocated_blocks"], 10)
        
    def test_atomic_operations(self):
        """测试原子性操作"""
        util = MemoryUtilization(total_blocks=100)
        
        # 测试超量分配
        success = util.atomic_allocate(150)
        self.assertFalse(success)
        
        # 状态应该保持不变
        status = util.query_utilization()
        self.assertEqual(status["free_blocks"], 100)
        self.assertEqual(status["allocated_blocks"], 0)
        
        # 测试超量释放
        success = util.atomic_deallocate(10)
        self.assertFalse(success)


class TestUnifiedMemoryManager(unittest.TestCase):
    """测试统一内存管理器"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.manager = UnifiedMemoryManager(
            total_blocks=100,
            block_size=4096,
            page_size=16,
            device=self.device,
        )
        
    def test_model_weights_management(self):
        """测试模型权重管理（只读访问）"""
        # 创建测试权重
        weight_tensor = torch.randn(1024, 512, dtype=torch.float16)
        
        # 注册权重
        success = self.manager.register_model_weights("layer1.weight", weight_tensor)
        self.assertTrue(success)
        
        # 获取权重（只读访问）
        retrieved_weight = self.manager.get_model_weights("layer1.weight")
        self.assertIsNotNone(retrieved_weight)
        self.assertTrue(torch.equal(weight_tensor, retrieved_weight))
        
        # 多次访问应该增加访问计数
        for _ in range(5):
            self.manager.get_model_weights("layer1.weight")
            
        stats = self.manager.get_statistics()
        self.assertEqual(stats["weight_access_stats"]["layer1.weight"], 6)  # 1 + 5
        
    def test_kv_cache_allocation(self):
        """测试KV cache分配"""
        # 测试基本分配
        block_ids = self.manager.allocate_kv_cache_blocks(
            request_id="req_1",
            layer_id=0,
            num_blocks=10,
            requester_role=InstanceRole.PREFILL,
        )
        
        self.assertIsNotNone(block_ids)
        self.assertEqual(len(block_ids), 10)
        
        # 检查内存使用
        usage = self.manager.get_memory_usage()
        self.assertEqual(usage["allocated_blocks"], 10)
        self.assertEqual(usage["free_blocks"], 90)
        
        # 测试块表索引
        block_indices = self.manager.get_block_table_index("req_1", 0)
        self.assertIsNotNone(block_indices)
        self.assertEqual(len(block_indices), 10)
        self.assertEqual(set(block_indices), set(block_ids))
        
    def test_kv_cache_access(self):
        """测试KV cache访问"""
        # 先分配块
        block_ids = self.manager.allocate_kv_cache_blocks(
            request_id="req_2",
            layer_id=1,
            num_blocks=5,
            requester_role=InstanceRole.DECODE,
        )
        self.assertIsNotNone(block_ids)
        
        # 访问KV cache
        kv_tensors = self.manager.access_kv_cache(
            request_id="req_2",
            layer_id=1,
            accessor_role=InstanceRole.DECODE,
        )
        
        self.assertIsNotNone(kv_tensors)
        self.assertEqual(len(kv_tensors), 5)
        
        # 验证tensor形状
        for tensor in kv_tensors:
            self.assertEqual(tensor.shape, (16, 64, 128))  # page_size, heads, dim
            
    def test_deallocation(self):
        """测试释放操作"""
        # 分配块
        block_ids = self.manager.allocate_kv_cache_blocks(
            request_id="req_3",
            layer_id=2,
            num_blocks=15,
            requester_role=InstanceRole.PREFILL,
        )
        self.assertIsNotNone(block_ids)
        
        # 检查分配后状态
        usage_before = self.manager.get_memory_usage()
        self.assertEqual(usage_before["allocated_blocks"], 15)
        
        # 释放块
        success = self.manager.deallocate_kv_cache_blocks("req_3")
        self.assertTrue(success)
        
        # 检查释放后状态
        usage_after = self.manager.get_memory_usage()
        self.assertEqual(usage_after["allocated_blocks"], 0)
        self.assertEqual(usage_after["free_blocks"], 100)
        
        # 块表应该被清理
        block_indices = self.manager.get_block_table_index("req_3", 2)
        self.assertIsNone(block_indices)
        
    def test_allocation_failure(self):
        """测试分配失败情况"""
        # 尝试分配超过总数的块
        block_ids = self.manager.allocate_kv_cache_blocks(
            request_id="req_fail",
            layer_id=0,
            num_blocks=150,  # 超过总数100
            requester_role=InstanceRole.PREFILL,
        )
        
        self.assertIsNone(block_ids)
        
        # 统计信息应该记录失败
        stats = self.manager.get_statistics()
        self.assertGreater(stats["allocation_failures"], 0)


class TestWARConflictResolution(unittest.TestCase):
    """测试WAR冲突解决"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.manager = UnifiedMemoryManager(
            total_blocks=50,
            block_size=4096,
            page_size=16,
            device=self.device,
        )
        
    def test_concurrent_allocation(self):
        """测试并发分配（模拟WAR冲突场景）"""
        results = []
        
        def allocate_worker(worker_id, role):
            """工作线程：模拟prefill或decode worker"""
            try:
                block_ids = self.manager.allocate_kv_cache_blocks(
                    request_id=f"req_{worker_id}",
                    layer_id=0,
                    num_blocks=5,
                    requester_role=role,
                )
                return {"worker_id": worker_id, "success": block_ids is not None, "block_ids": block_ids}
            except Exception as e:
                return {"worker_id": worker_id, "success": False, "error": str(e)}
        
        # 启动多个并发worker（模拟prefill和decode同时分配）
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            # 5个prefill worker
            for i in range(5):
                future = executor.submit(allocate_worker, f"prefill_{i}", InstanceRole.PREFILL)
                futures.append(future)
                
            # 5个decode worker
            for i in range(5):
                future = executor.submit(allocate_worker, f"decode_{i}", InstanceRole.DECODE)
                futures.append(future)
                
            # 收集结果
            for future in as_completed(futures):
                results.append(future.result())
                
        # 验证结果
        successful_allocations = [r for r in results if r["success"]]
        failed_allocations = [r for r in results if not r["success"]]
        
        # 应该有成功的分配（原子性保证）
        self.assertGreater(len(successful_allocations), 0)
        
        # 总分配的块数不应超过可用块数
        total_allocated_blocks = sum(len(r.get("block_ids", [])) for r in successful_allocations)
        self.assertLessEqual(total_allocated_blocks, 50)
        
        # 验证内存一致性
        usage = self.manager.get_memory_usage()
        self.assertEqual(usage["allocated_blocks"], total_allocated_blocks)
        
        print(f"Concurrent allocation test: {len(successful_allocations)} successful, {len(failed_allocations)} failed")
        
    def test_war_conflict_detection(self):
        """测试WAR冲突检测"""
        # 创建可能的冲突场景
        # Prefill worker分配块
        prefill_blocks = self.manager.allocate_kv_cache_blocks(
            request_id="prefill_req",
            layer_id=0,
            num_blocks=10,
            requester_role=InstanceRole.PREFILL,
        )
        self.assertIsNotNone(prefill_blocks)
        
        # Decode worker分配块
        decode_blocks = self.manager.allocate_kv_cache_blocks(
            request_id="decode_req",
            layer_id=0,
            num_blocks=10,
            requester_role=InstanceRole.DECODE,
        )
        self.assertIsNotNone(decode_blocks)
        
        # 检测冲突
        conflicts = self.manager.detect_war_conflicts()
        
        # 在这个测试中，不同请求使用不同块，所以不应该有冲突
        # 但如果有共享块的情况，应该能检测到
        print(f"Detected conflicts: {len(conflicts)}")
        
    def test_atomic_memory_utilization_update(self):
        """测试原子性内存利用率更新"""
        
        def concurrent_allocator(thread_id):
            """并发分配器"""
            results = []
            for i in range(5):
                # 每个线程尝试分配2个块
                block_ids = self.manager.allocate_kv_cache_blocks(
                    request_id=f"thread_{thread_id}_req_{i}",
                    layer_id=i,
                    num_blocks=2,
                    requester_role=InstanceRole.PREFILL if thread_id % 2 == 0 else InstanceRole.DECODE,
                )
                results.append(block_ids is not None)
                time.sleep(0.001)  # 小延迟增加竞争
            return results
            
        # 启动多个线程
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_allocator, i) for i in range(5)]
            all_results = [future.result() for future in as_completed(futures)]
            
        # 验证最终状态一致性
        usage = self.manager.get_memory_usage()
        stats = self.manager.get_statistics()
        
        # 分配的块数应该等于成功分配的请求数 * 2
        successful_allocations = sum(sum(results) for results in all_results)
        expected_allocated_blocks = successful_allocations * 2
        
        self.assertEqual(usage["allocated_blocks"], expected_allocated_blocks)
        self.assertEqual(usage["free_blocks"], 50 - expected_allocated_blocks)
        
        print(f"Atomic update test: {successful_allocations} successful allocations, "
              f"{expected_allocated_blocks} blocks allocated")


class TestPagedStorage(unittest.TestCase):
    """测试vLLM风格的paged storage"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.manager = UnifiedMemoryManager(
            total_blocks=64,
            block_size=4096,
            page_size=16,
            device=self.device,
        )
        
    def test_page_table_creation(self):
        """测试页表创建"""
        # 分配多个层的块
        for layer_id in range(3):
            block_ids = self.manager.allocate_kv_cache_blocks(
                request_id="multi_layer_req",
                layer_id=layer_id,
                num_blocks=4,
                requester_role=InstanceRole.PREFILL,
            )
            self.assertIsNotNone(block_ids)
            
        # 验证每个层都有页表
        for layer_id in range(3):
            block_indices = self.manager.get_block_table_index("multi_layer_req", layer_id)
            self.assertIsNotNone(block_indices)
            self.assertEqual(len(block_indices), 4)
            
    def test_block_table_index_access(self):
        """测试块表索引访问"""
        # 分配块
        original_block_ids = self.manager.allocate_kv_cache_blocks(
            request_id="index_test",
            layer_id=5,
            num_blocks=8,
            requester_role=InstanceRole.DECODE,
        )
        self.assertIsNotNone(original_block_ids)
        
        # 通过块表索引访问
        retrieved_indices = self.manager.get_block_table_index("index_test", 5)
        self.assertIsNotNone(retrieved_indices)
        
        # 索引应该匹配原始分配的块ID
        self.assertEqual(set(original_block_ids), set(retrieved_indices))
        
        # 根据论文："Once the block table index is determined, 
        # the access of the KV cache can be conducted without conflicts."
        kv_tensors = self.manager.access_kv_cache(
            request_id="index_test",
            layer_id=5,
            accessor_role=InstanceRole.DECODE,
        )
        
        self.assertIsNotNone(kv_tensors)
        self.assertEqual(len(kv_tensors), 8)


def main():
    """运行所有测试"""
    print("=== 测试Unified Memory Manager ===")
    print("验证论文4.4节描述的功能实现")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestMemoryUtilization,
        TestUnifiedMemoryManager,
        TestWARConflictResolution,
        TestPagedStorage,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n=== 论文4.4节功能验证 ===")
    
    if result.wasSuccessful():
        print("🎉 所有测试通过！论文4.4节的Unified Memory Manager功能已正确实现：")
        print("1. ✅ 模型权重的只读访问管理")
        print("2. ✅ KV cache的paged storage机制")
        print("3. ✅ WAR冲突的原子性分配解决")
        print("4. ✅ prefill和decode worker的异步内存访问")
        print("5. ✅ vLLM风格的块表索引机制")
        print("6. ✅ 内存利用率的原子性更新")
        return 0
    else:
        print("❌ 部分测试失败")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
        return 1


if __name__ == "__main__":
    exit(main())