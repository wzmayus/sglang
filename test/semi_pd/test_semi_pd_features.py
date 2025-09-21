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
"""Tests for Semi-PD new features."""

import time
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.semi_pd.process_rotation_manager import (
    ProcessRotationManager,
    SMAllocation,
)
from sglang.semi_pd.semi_pd_coordinator import (
    SemiPDCoordinator,
    create_semi_pd_coordinator,
)
from sglang.semi_pd.slo_aware_resource_manager import (
    SLOAwareResourceManager,
    SLOTarget,
    WorkloadMetrics,
)
from sglang.semi_pd.unified_storage_manager import UnifiedStorageManager
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import PortArgs, ServerArgs


class TestSMAllocation(unittest.TestCase):
    """测试SM分配配置"""
    
    def test_valid_allocation(self):
        """测试有效的SM分配"""
        allocation = SMAllocation(prefill_percentage=70, decode_percentage=30)
        self.assertEqual(allocation.prefill_percentage, 70)
        self.assertEqual(allocation.decode_percentage, 30)
        
    def test_invalid_allocation(self):
        """测试无效的SM分配"""
        with self.assertRaises(ValueError):
            SMAllocation(prefill_percentage=80, decode_percentage=50)  # 总和超过100%


class TestUnifiedStorageManager(unittest.TestCase):
    """测试统一存储管理器"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.storage_manager = UnifiedStorageManager(
            total_blocks=100,
            block_size=4096,
            device=self.device,
        )
        
    def test_block_allocation(self):
        """测试块分配"""
        # 分配块
        block_ids = self.storage_manager.allocate_blocks(
            request_id="test_req_1",
            num_blocks=10,
            requester_role=InstanceRole.PREFILL,
        )
        
        self.assertIsNotNone(block_ids)
        self.assertEqual(len(block_ids), 10)
        
        # 检查内存使用
        usage = self.storage_manager.get_memory_usage()
        self.assertEqual(usage["allocated_blocks"], 10)
        
    def test_block_deallocation(self):
        """测试块释放"""
        # 先分配
        block_ids = self.storage_manager.allocate_blocks("test_req_2", 5)
        self.assertIsNotNone(block_ids)
        
        # 再释放
        success = self.storage_manager.deallocate_blocks("test_req_2")
        self.assertTrue(success)
        
        # 检查内存使用
        usage = self.storage_manager.get_memory_usage()
        self.assertEqual(usage["allocated_blocks"], 0)
        
    def test_insufficient_blocks(self):
        """测试块不足的情况"""
        # 尝试分配超过总数的块
        block_ids = self.storage_manager.allocate_blocks("test_req_3", 200)
        self.assertIsNone(block_ids)
        
    def test_block_sharing(self):
        """测试块共享（prefix caching）"""
        # 启用prefix caching
        storage_manager = UnifiedStorageManager(
            total_blocks=100,
            block_size=4096,
            device=self.device,
            enable_prefix_caching=True,
        )
        
        # 分配块给第一个请求
        block_ids = storage_manager.allocate_blocks("req_1", 5)
        self.assertIsNotNone(block_ids)
        
        # 共享给第二个请求
        success = storage_manager.share_blocks("req_1", "req_2")
        self.assertTrue(success)
        
        # 检查两个请求都有分配记录
        handles_1 = storage_manager.get_block_handles("req_1")
        handles_2 = storage_manager.get_block_handles("req_2")
        self.assertIsNotNone(handles_1)
        self.assertIsNotNone(handles_2)


class TestSLOAwareResourceManager(unittest.TestCase):
    """测试SLO感知资源管理器"""
    
    def setUp(self):
        self.slo_target = SLOTarget(ttft_target_ms=100.0, tpot_target_ms=50.0)
        
        # Mock进程轮换管理器
        self.mock_rotation_manager = MagicMock()
        self.mock_rotation_manager.get_status.return_value = {
            "current_sm_allocation": {
                "prefill_percentage": 70,
                "decode_percentage": 30,
            }
        }
        
        self.slo_manager = SLOAwareResourceManager(
            slo_target=self.slo_target,
            process_rotation_manager=self.mock_rotation_manager,
            window_size_seconds=10.0,
            adjustment_cooldown_seconds=5.0,
        )
        
    def test_metrics_reporting(self):
        """测试指标报告"""
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
        
        self.slo_manager.report_metrics(metrics)
        
        # 检查指标是否被记录
        self.assertEqual(len(self.slo_manager.metrics_window), 1)
        
    def test_slo_violation_detection(self):
        """测试SLO违反检测"""
        # 报告违反SLO的指标
        bad_metrics = WorkloadMetrics(
            timestamp=time.time(),
            prefill_queue_length=20,
            decode_queue_length=30,
            prefill_throughput=500.0,
            decode_throughput=800.0,
            ttft_p95=150.0,  # 超过目标100ms
            tpot_p95=80.0,   # 超过目标50ms
            prefill_utilization=0.9,
            decode_utilization=0.95,
        )
        
        self.slo_manager.report_metrics(bad_metrics)
        
        # 检查SLO违反统计
        compliance = self.slo_manager.get_slo_compliance_rate()
        self.assertLess(compliance["ttft_compliance"], 1.0)
        self.assertLess(compliance["tpot_compliance"], 1.0)


class TestProcessRotationManager(unittest.TestCase):
    """测试进程轮换管理器"""
    
    def setUp(self):
        # Mock server args
        self.server_args = MagicMock()
        self.port_args = MagicMock()
        
        self.initial_allocation = SMAllocation(
            prefill_percentage=70,
            decode_percentage=30,
        )
        
    @patch('sglang.semi_pd.process_rotation_manager.mp.Process')
    def test_sm_reallocation_request(self, mock_process):
        """测试SM重新分配请求"""
        # Mock进程
        mock_process_instance = MagicMock()
        mock_process_instance.is_alive.return_value = True
        mock_process.return_value = mock_process_instance
        
        rotation_manager = ProcessRotationManager(
            server_args=self.server_args,
            port_args=self.port_args,
            initial_sm_allocation=self.initial_allocation,
        )
        
        # 请求新的分配
        new_allocation = SMAllocation(prefill_percentage=60, decode_percentage=40)
        
        with patch.object(rotation_manager, '_start_standby_processes', return_value=True):
            success = rotation_manager.request_sm_reallocation(new_allocation)
            self.assertTrue(success)
            
        # 检查状态
        status = rotation_manager.get_status()
        self.assertTrue(status["switch_requested"])


class TestSemiPDCoordinator(unittest.TestCase):
    """测试Semi-PD协调器"""
    
    def setUp(self):
        self.server_args = MagicMock()
        self.server_args.enable_prefix_caching = False
        self.port_args = MagicMock()
        self.device = torch.device("cpu")
        
    @patch('sglang.semi_pd.semi_pd_coordinator.ProcessRotationManager')
    @patch('sglang.semi_pd.semi_pd_coordinator.SLOAwareResourceManager')
    def test_coordinator_initialization(self, mock_slo_manager, mock_rotation_manager):
        """测试协调器初始化"""
        initial_allocation = SMAllocation(prefill_percentage=70, decode_percentage=30)
        slo_target = SLOTarget(ttft_target_ms=100.0, tpot_target_ms=50.0)
        
        coordinator = SemiPDCoordinator(
            server_args=self.server_args,
            port_args=self.port_args,
            initial_sm_allocation=initial_allocation,
            slo_target=slo_target,
            total_kv_blocks=1000,
            kv_block_size=4096,
            device=self.device,
        )
        
        self.assertIsNotNone(coordinator.process_rotation_manager)
        self.assertIsNotNone(coordinator.storage_manager)
        self.assertIsNotNone(coordinator.slo_manager)
        
    def test_create_semi_pd_coordinator(self):
        """测试便利函数"""
        with patch('sglang.semi_pd.semi_pd_coordinator.ProcessRotationManager'), \
             patch('sglang.semi_pd.semi_pd_coordinator.SLOAwareResourceManager'):
            
            coordinator = create_semi_pd_coordinator(
                server_args=self.server_args,
                port_args=self.port_args,
                prefill_sm_percentage=60,
                decode_sm_percentage=40,
            )
            
            self.assertIsNotNone(coordinator)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    @patch('sglang.semi_pd.process_rotation_manager.mp.Process')
    def test_end_to_end_workflow(self, mock_process):
        """测试端到端工作流"""
        # Mock进程
        mock_process_instance = MagicMock()
        mock_process_instance.is_alive.return_value = True
        mock_process.return_value = mock_process_instance
        
        # 创建协调器
        server_args = MagicMock()
        server_args.enable_prefix_caching = False
        port_args = MagicMock()
        
        with patch('sglang.semi_pd.semi_pd_coordinator.ProcessRotationManager') as mock_rotation, \
             patch('sglang.semi_pd.semi_pd_coordinator.SLOAwareResourceManager') as mock_slo:
            
            # Mock返回值
            mock_rotation_instance = MagicMock()
            mock_rotation.return_value = mock_rotation_instance
            
            mock_slo_instance = MagicMock()
            mock_slo.return_value = mock_slo_instance
            
            coordinator = create_semi_pd_coordinator(
                server_args=server_args,
                port_args=port_args,
            )
            
            # 测试KV块分配
            handles = coordinator.allocate_kv_blocks("test_req", 5, InstanceRole.PREFILL)
            self.assertIsNotNone(handles)
            
            # 测试指标报告
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
            
            coordinator.report_workload_metrics(metrics)
            
            # 测试资源调整
            new_allocation = SMAllocation(prefill_percentage=60, decode_percentage=40)
            success = coordinator.request_resource_adjustment(new_allocation)
            
            # 测试状态获取
            status = coordinator.get_comprehensive_status()
            self.assertIn("coordinator", status)
            
            # 测试健康检查
            health = coordinator.health_check()
            self.assertIn("healthy", health)


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2)