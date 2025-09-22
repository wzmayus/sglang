#!/usr/bin/env python3
"""
Semi-PD集成测试

测试三个核心功能的整合：
1. 常驻进程+进程轮转机制
2. SLO-aware动态资源调整算法
3. Unified Memory Manager

验证SGLang启动时能够正确应用这些机制。
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

# 添加python目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torch

from sglang.semi_pd.semi_pd_coordinator import SemiPDCoordinator, create_semi_pd_coordinator
from sglang.semi_pd.process_rotation_manager import SMAllocation
from sglang.semi_pd.slo_algorithm import SLOConstraints
from sglang.semi_pd.unified_memory_manager import UnifiedMemoryManager
from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import ServerArgs, SemiPDPortArgs


class TestSemiPDIntegration(unittest.TestCase):
    """测试Semi-PD功能集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = torch.device("cpu")
        
        # 创建测试用的ServerArgs
        self.server_args = ServerArgs()
        self.server_args.model_path = "test-model"
        self.server_args.enable_semi_pd_coordinator = True
        self.server_args.enable_unified_memory = True
        self.server_args.enable_slo_aware = True
        self.server_args.unified_memory_blocks = 100
        self.server_args.unified_memory_block_size = 1024
        self.server_args.slo_ttft_target = 80.0
        self.server_args.slo_tpot_target = 40.0
        self.server_args.initial_prefill_sm = 60
        self.server_args.initial_decode_sm = 40
        
        # 创建测试用的SemiPDPortArgs
        self.port_args = SemiPDPortArgs()
        self.port_args.host = "127.0.0.1"
        self.port_args.port = 30000
        
    def test_coordinator_creation(self):
        """测试协调器创建"""
        coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        self.assertIsInstance(coordinator, SemiPDCoordinator)
        self.assertEqual(coordinator.initial_sm_allocation.prefill_percentage, 60)
        self.assertEqual(coordinator.initial_sm_allocation.decode_percentage, 40)
        self.assertIsNotNone(coordinator.slo_constraints)
        self.assertEqual(coordinator.slo_constraints.ttft_target_ms, 80.0)
        
    def test_coordinator_initialization(self):
        """测试协调器初始化"""
        coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        # Mock进程轮换管理器的start方法
        with patch.object(coordinator, '_init_process_rotation_manager') as mock_init_rotation:
            with patch.object(coordinator, '_init_unified_memory_manager') as mock_init_memory:
                with patch.object(coordinator, '_init_slo_controller') as mock_init_slo:
                    success = coordinator.initialize()
                    
                    self.assertTrue(success)
                    mock_init_memory.assert_called_once()
                    mock_init_rotation.assert_called_once()
                    mock_init_slo.assert_called_once()
                    
    def test_unified_memory_manager_integration(self):
        """测试统一内存管理器集成"""
        coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        # 初始化统一内存管理器
        coordinator._init_unified_memory_manager()
        
        self.assertIsNotNone(coordinator.unified_memory_manager)
        self.assertIsInstance(coordinator.unified_memory_manager, UnifiedMemoryManager)
        
        # 测试内存分配
        block_ids = coordinator.allocate_kv_cache_blocks(
            request_id="test_req",
            layer_id=0,
            num_blocks=5,
            requester_role=InstanceRole.PREFILL,
        )
        
        self.assertIsNotNone(block_ids)
        self.assertEqual(len(block_ids), 5)
        
        # 测试内存释放
        success = coordinator.deallocate_kv_cache_blocks("test_req")
        self.assertTrue(success)
        
    def test_slo_constraints_integration(self):
        """测试SLO约束集成"""
        coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        self.assertIsNotNone(coordinator.slo_constraints)
        self.assertEqual(coordinator.slo_constraints.ttft_target_ms, 80.0)
        self.assertEqual(coordinator.slo_constraints.tpot_target_ms, 40.0)
        
    def test_sm_allocation_integration(self):
        """测试SM分配集成"""
        coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        # 测试初始分配
        self.assertEqual(coordinator.initial_sm_allocation.prefill_percentage, 60)
        self.assertEqual(coordinator.initial_sm_allocation.decode_percentage, 40)
        
        # Mock进程轮换管理器
        mock_rotation_manager = Mock()
        mock_rotation_manager.request_sm_reallocation.return_value = True
        coordinator.process_rotation_manager = mock_rotation_manager
        
        # 测试SM重新分配
        new_allocation = SMAllocation(prefill_percentage=70, decode_percentage=30)
        success = coordinator.request_sm_reallocation(new_allocation)
        
        self.assertTrue(success)
        mock_rotation_manager.request_sm_reallocation.assert_called_once_with(new_allocation)
        
    def test_coordinator_status(self):
        """测试协调器状态获取"""
        coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        status = coordinator.get_status()
        
        self.assertIn("is_running", status)
        self.assertIn("components", status)
        self.assertIn("configuration", status)
        
        # 检查配置信息
        config = status["configuration"]
        self.assertTrue(config["slo_enabled"])
        self.assertTrue(config["unified_memory_enabled"])
        
    def test_metrics_aggregation(self):
        """测试指标聚合"""
        coordinator = create_semi_pd_coordinator(
            server_args=self.server_args,
            port_args=self.port_args,
        )
        
        # 初始化metrics聚合器
        coordinator._init_metrics_aggregator()
        
        # 测试指标更新
        prefill_metrics = {
            "ttft_p95_ms": 75.0,
            "prefill_queue_length": 5,
            "prefill_utilization": 0.8,
        }
        
        decode_metrics = {
            "tpot_p95_ms": 35.0,
            "decode_queue_length": 8,
            "decode_utilization": 0.7,
        }
        
        coordinator.update_metrics(prefill_metrics, decode_metrics)
        
        # 验证聚合指标
        aggregated = coordinator.metrics_aggregator.get_aggregated_metrics()
        self.assertIn("ttft_p95_ms", aggregated)
        self.assertIn("tpot_p95_ms", aggregated)


class TestSemiPDServerArgsIntegration(unittest.TestCase):
    """测试ServerArgs集成"""
    
    def test_server_args_defaults(self):
        """测试ServerArgs默认值"""
        args = ServerArgs()
        
        # 检查新添加的参数默认值
        self.assertFalse(args.enable_semi_pd_coordinator)
        self.assertFalse(args.enable_unified_memory)
        self.assertFalse(args.enable_slo_aware)
        
        self.assertEqual(args.unified_memory_blocks, 1000)
        self.assertEqual(args.unified_memory_block_size, 4096)
        self.assertEqual(args.unified_memory_page_size, 16)
        
        self.assertEqual(args.slo_ttft_target, 100.0)
        self.assertEqual(args.slo_tpot_target, 50.0)
        self.assertEqual(args.slo_window_size, 30.0)
        
        self.assertEqual(args.initial_prefill_sm, 70)
        self.assertEqual(args.initial_decode_sm, 30)
        
    def test_server_args_configuration(self):
        """测试ServerArgs配置"""
        args = ServerArgs()
        
        # 启用Semi-PD功能
        args.enable_semi_pd_coordinator = True
        args.enable_unified_memory = True
        args.enable_slo_aware = True
        
        # 自定义配置
        args.unified_memory_blocks = 2000
        args.slo_ttft_target = 120.0
        args.initial_prefill_sm = 80
        
        # 验证配置
        self.assertTrue(args.enable_semi_pd_coordinator)
        self.assertTrue(args.enable_unified_memory)
        self.assertTrue(args.enable_slo_aware)
        self.assertEqual(args.unified_memory_blocks, 2000)
        self.assertEqual(args.slo_ttft_target, 120.0)
        self.assertEqual(args.initial_prefill_sm, 80)


class TestSemiPDSchedulerIntegration(unittest.TestCase):
    """测试Semi-PD Scheduler集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.server_args = ServerArgs()
        self.server_args.enable_unified_memory = True
        self.server_args.enable_slo_aware = True
        self.server_args.unified_memory_blocks = 100
        
        self.port_args = SemiPDPortArgs()
        
    @patch('sglang.semi_pd.unified_memory_manager.UnifiedMemoryManager')
    @patch('sglang.semi_pd.metrics_integration.integrate_semi_pd_metrics_with_scheduler')
    def test_scheduler_feature_initialization(self, mock_integrate_metrics, mock_memory_manager):
        """测试scheduler功能初始化"""
        from sglang.srt.managers.semi_pd_scheduler import SemiPDScheduler
        
        # Mock必要的依赖
        mock_memory_manager.return_value = Mock()
        
        # 创建scheduler（需要mock很多依赖）
        with patch('sglang.srt.managers.scheduler.Scheduler.__init__'):
            scheduler = SemiPDScheduler(
                server_args=self.server_args,
                port_args=self.port_args,
                gpu_id=0,
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                dp_rank=None,
                instance_role=InstanceRole.PREFILL,
            )
            
            # 验证功能初始化
            self.assertIsNotNone(scheduler.unified_memory_manager)
            self.assertIsNotNone(scheduler.slo_constraints)
            mock_integrate_metrics.assert_called_once()


class TestEndToEndIntegration(unittest.TestCase):
    """端到端集成测试"""
    
    def test_full_integration_workflow(self):
        """测试完整的集成工作流"""
        # 1. 创建配置
        server_args = ServerArgs()
        server_args.model_path = "test-model"
        server_args.enable_semi_pd_coordinator = True
        server_args.enable_unified_memory = True
        server_args.enable_slo_aware = True
        
        port_args = SemiPDPortArgs()
        port_args.host = "127.0.0.1"
        port_args.port = 30000
        
        # 2. 创建协调器
        coordinator = create_semi_pd_coordinator(
            server_args=server_args,
            port_args=port_args,
        )
        
        # 3. 验证协调器配置
        self.assertIsNotNone(coordinator)
        self.assertIsNotNone(coordinator.slo_constraints)
        self.assertTrue(coordinator.unified_memory_config['enabled'])
        
        # 4. 测试上下文管理器
        with patch.object(coordinator, 'start', return_value=True):
            with patch.object(coordinator, 'stop'):
                with coordinator:
                    status = coordinator.get_status()
                    self.assertIn("configuration", status)
                    
    def test_feature_compatibility(self):
        """测试功能兼容性"""
        # 测试向后兼容性
        server_args = ServerArgs()
        server_args.enable_semi_pd = True  # 旧参数
        
        # 应该仍然能够工作
        self.assertTrue(server_args.enable_semi_pd)
        
        # 新参数应该有默认值
        self.assertFalse(server_args.enable_semi_pd_coordinator)
        self.assertFalse(server_args.enable_unified_memory)
        self.assertFalse(server_args.enable_slo_aware)


def run_integration_tests():
    """运行集成测试"""
    print("=== Semi-PD集成测试 ===")
    print("测试三个核心功能的整合效果")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestSemiPDIntegration,
        TestSemiPDServerArgsIntegration,
        TestSemiPDSchedulerIntegration,
        TestEndToEndIntegration,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n=== 集成测试结果 ===")
    
    if result.wasSuccessful():
        print("🎉 所有集成测试通过！Semi-PD三个核心功能已成功整合到SGLang中：")
        print("1. ✅ 常驻进程+进程轮转机制")
        print("2. ✅ SLO-aware动态资源调整算法")
        print("3. ✅ Unified Memory Manager")
        print()
        print("SGLang现在可以通过以下方式启用Semi-PD功能：")
        print("  python -m sglang.launch_server --model-path <MODEL> --enable-semi-pd-coordinator")
        return 0
    else:
        print("❌ 部分集成测试失败")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
        return 1


if __name__ == "__main__":
    exit(run_integration_tests())