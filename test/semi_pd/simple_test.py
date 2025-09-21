#!/usr/bin/env python3
"""
简化的Semi-PD功能测试

这个测试文件验证Semi-PD新功能的基本实现是否正确。
"""

import sys
import os

# 添加python目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import time
import torch
from unittest.mock import MagicMock

def test_imports():
    """测试模块导入"""
    print("Testing imports...")
    
    try:
        from sglang.semi_pd.utils import InstanceRole, SMAllocation
        from sglang.semi_pd.unified_storage_manager import UnifiedStorageManager
        from sglang.semi_pd.process_rotation_manager import ProcessRotationManager, SMAllocation
        from sglang.semi_pd.slo_aware_resource_manager import SLOTarget, WorkloadMetrics
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_sm_allocation():
    """测试SM分配配置"""
    print("Testing SM allocation...")
    
    try:
        from sglang.semi_pd.process_rotation_manager import SMAllocation
        
        # 测试有效分配
        allocation = SMAllocation(prefill_percentage=70, decode_percentage=30)
        assert allocation.prefill_percentage == 70
        assert allocation.decode_percentage == 30
        
        # 测试无效分配
        try:
            SMAllocation(prefill_percentage=80, decode_percentage=50)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
            
        print("✓ SM allocation tests passed")
        return True
    except Exception as e:
        print(f"✗ SM allocation test failed: {e}")
        return False

def test_storage_manager():
    """测试存储管理器"""
    print("Testing storage manager...")
    
    try:
        from sglang.semi_pd.unified_storage_manager import UnifiedStorageManager
        from sglang.semi_pd.utils import InstanceRole
        
        device = torch.device("cpu")
        storage_manager = UnifiedStorageManager(
            total_blocks=100,
            block_size=4096,
            device=device,
        )
        
        # 测试块分配
        block_ids = storage_manager.allocate_blocks(
            request_id="test_req_1",
            num_blocks=10,
            requester_role=InstanceRole.PREFILL,
        )
        
        assert block_ids is not None
        assert len(block_ids) == 10
        
        # 测试内存使用
        usage = storage_manager.get_memory_usage()
        assert usage["allocated_blocks"] == 10
        
        # 测试块释放
        success = storage_manager.deallocate_blocks("test_req_1")
        assert success
        
        usage = storage_manager.get_memory_usage()
        assert usage["allocated_blocks"] == 0
        
        print("✓ Storage manager tests passed")
        return True
    except Exception as e:
        print(f"✗ Storage manager test failed: {e}")
        return False

def test_slo_manager():
    """测试SLO管理器"""
    print("Testing SLO manager...")
    
    try:
        from sglang.semi_pd.slo_aware_resource_manager import (
            SLOAwareResourceManager, SLOTarget, WorkloadMetrics
        )
        
        slo_target = SLOTarget(ttft_target_ms=100.0, tpot_target_ms=50.0)
        
        # Mock进程轮换管理器
        mock_rotation_manager = MagicMock()
        mock_rotation_manager.get_status.return_value = {
            "current_sm_allocation": {
                "prefill_percentage": 70,
                "decode_percentage": 30,
            }
        }
        
        slo_manager = SLOAwareResourceManager(
            slo_target=slo_target,
            process_rotation_manager=mock_rotation_manager,
            window_size_seconds=10.0,
            adjustment_cooldown_seconds=5.0,
        )
        
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
        
        slo_manager.report_metrics(metrics)
        assert len(slo_manager.metrics_window) == 1
        
        print("✓ SLO manager tests passed")
        return True
    except Exception as e:
        print(f"✗ SLO manager test failed: {e}")
        return False

def test_coordinator():
    """测试协调器"""
    print("Testing coordinator...")
    
    try:
        from sglang.semi_pd.semi_pd_coordinator import create_semi_pd_coordinator
        from unittest.mock import patch
        
        # Mock server args
        server_args = MagicMock()
        server_args.enable_prefix_caching = False
        port_args = MagicMock()
        
        with patch('sglang.semi_pd.semi_pd_coordinator.ProcessRotationManager') as mock_rotation, \
             patch('sglang.semi_pd.semi_pd_coordinator.SLOAwareResourceManager') as mock_slo:
            
            coordinator = create_semi_pd_coordinator(
                server_args=server_args,
                port_args=port_args,
            )
            
            assert coordinator is not None
            
        print("✓ Coordinator tests passed")
        return True
    except Exception as e:
        print(f"✗ Coordinator test failed: {e}")
        return False

def main():
    """运行所有测试"""
    print("=== Semi-PD功能测试 ===")
    
    tests = [
        test_imports,
        test_sm_allocation,
        test_storage_manager,
        test_slo_manager,
        test_coordinator,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    exit(main())