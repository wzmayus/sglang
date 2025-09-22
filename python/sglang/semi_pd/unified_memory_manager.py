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
Unified Memory Manager for Semi-PD

根据论文4.4节实现统一内存管理器，主要解决：
1. 模型权重的只读访问
2. KV cache的paged storage管理
3. WAR冲突的原子性分配
4. prefill和decode worker的异步内存访问
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import numpy as np

from sglang.semi_pd.utils import InstanceRole

logger = logging.getLogger(__name__)


class BlockState(Enum):
    """KV cache块状态"""
    FREE = "free"           # 空闲块
    ALLOCATED = "allocated" # 已分配
    IN_USE = "in_use"      # 使用中
    RESERVED = "reserved"   # 预留块


@dataclass
class BlockInfo:
    """KV cache块信息"""
    block_id: int
    state: BlockState
    request_id: Optional[str] = None
    layer_id: Optional[int] = None
    allocated_time: float = 0.0
    last_access_time: float = 0.0
    reference_count: int = 0
    allocator_role: Optional[InstanceRole] = None


@dataclass
class PageTableEntry:
    """页表项"""
    virtual_page_id: int
    physical_block_id: int
    request_id: str
    layer_id: int
    is_valid: bool = True


@dataclass
class AllocationRequest:
    """分配请求"""
    request_id: str
    layer_id: int
    num_blocks: int
    requester_role: InstanceRole
    priority: int = 0
    timestamp: float = 0.0


class MemoryUtilization:
    """内存利用率管理"""
    
    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        self.allocated_blocks = 0
        self.free_blocks = total_blocks
        self.reserved_blocks = 0
        self.last_update_time = time.time()
        
        # 原子操作锁 - 解决WAR冲突
        self._lock = threading.RLock()
        
    def query_utilization(self) -> Dict[str, float]:
        """查询内存利用率（原子操作第一步）"""
        with self._lock:
            return {
                "total_blocks": self.total_blocks,
                "allocated_blocks": self.allocated_blocks,
                "free_blocks": self.free_blocks,
                "reserved_blocks": self.reserved_blocks,
                "utilization_ratio": self.allocated_blocks / self.total_blocks,
                "free_ratio": self.free_blocks / self.total_blocks,
                "query_time": time.time(),
            }
            
    def atomic_allocate(self, num_blocks: int) -> bool:
        """原子性分配（解决WAR冲突）"""
        with self._lock:
            # 检查是否有足够的空闲块
            if self.free_blocks >= num_blocks:
                # 原子性更新内存利用率
                self.free_blocks -= num_blocks
                self.allocated_blocks += num_blocks
                self.last_update_time = time.time()
                return True
            else:
                return False
                
    def atomic_deallocate(self, num_blocks: int) -> bool:
        """原子性释放"""
        with self._lock:
            if self.allocated_blocks >= num_blocks:
                self.allocated_blocks -= num_blocks
                self.free_blocks += num_blocks
                self.last_update_time = time.time()
                return True
            else:
                logger.warning(f"Attempting to deallocate {num_blocks} blocks, but only {self.allocated_blocks} allocated")
                return False
                
    def reserve_blocks(self, num_blocks: int) -> bool:
        """预留块"""
        with self._lock:
            if self.free_blocks >= num_blocks:
                self.free_blocks -= num_blocks
                self.reserved_blocks += num_blocks
                return True
            return False
            
    def release_reserved_blocks(self, num_blocks: int) -> bool:
        """释放预留块"""
        with self._lock:
            if self.reserved_blocks >= num_blocks:
                self.reserved_blocks -= num_blocks
                self.free_blocks += num_blocks
                return True
            return False


class UnifiedMemoryManager:
    """
    统一内存管理器
    
    根据论文4.4节实现，主要功能：
    1. 管理模型权重的只读访问
    2. 管理KV cache的paged storage
    3. 解决prefill和decode worker的WAR冲突
    4. 提供原子性的内存分配操作
    """
    
    def __init__(
        self,
        total_blocks: int,
        block_size: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        enable_prefix_caching: bool = False,
    ):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.enable_prefix_caching = enable_prefix_caching
        
        # 内存利用率管理
        self.memory_utilization = MemoryUtilization(total_blocks)
        
        # 块管理
        self.blocks: Dict[int, BlockInfo] = {}
        self.free_block_ids: Set[int] = set(range(total_blocks))
        self.allocated_blocks: Dict[str, List[int]] = {}  # request_id -> block_ids
        
        # 页表管理（vLLM风格的paged storage）
        self.page_tables: Dict[str, Dict[int, List[PageTableEntry]]] = {}  # request_id -> layer_id -> entries
        
        # 同步控制 - 解决WAR冲突的关键
        self.allocation_lock = threading.RLock()  # 分配锁
        self.access_lock = threading.RLock()      # 访问锁
        self.page_table_lock = threading.RLock()  # 页表锁
        
        # 模型权重管理（只读）
        self.model_weights: Dict[str, torch.Tensor] = {}
        self.weight_access_count: Dict[str, int] = {}
        
        # 统计信息
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "allocation_failures": 0,
            "war_conflicts_prevented": 0,
            "peak_utilization": 0.0,
            "avg_allocation_time": 0.0,
        }
        
        # 初始化块池
        self._initialize_blocks()
        
        logger.info(f"Unified memory manager initialized: {total_blocks} blocks, {block_size} bytes per block")
        
    def _initialize_blocks(self):
        """初始化块池"""
        for block_id in range(self.total_blocks):
            self.blocks[block_id] = BlockInfo(
                block_id=block_id,
                state=BlockState.FREE,
            )
            
    def register_model_weights(self, weight_name: str, weight_tensor: torch.Tensor) -> bool:
        """注册模型权重（只读访问）"""
        try:
            # 模型权重是只读的，可以直接共享
            self.model_weights[weight_name] = weight_tensor
            self.weight_access_count[weight_name] = 0
            
            logger.info(f"Registered model weight: {weight_name}, shape: {weight_tensor.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model weight {weight_name}: {e}")
            return False
            
    def get_model_weights(self, weight_name: str) -> Optional[torch.Tensor]:
        """获取模型权重（只读访问）"""
        if weight_name in self.model_weights:
            self.weight_access_count[weight_name] += 1
            return self.model_weights[weight_name]
        else:
            logger.warning(f"Model weight not found: {weight_name}")
            return None
            
    def allocate_kv_cache_blocks(
        self,
        request_id: str,
        layer_id: int,
        num_blocks: int,
        requester_role: InstanceRole,
        priority: int = 0,
    ) -> Optional[List[int]]:
        """
        分配KV cache块（原子性操作，解决WAR冲突）
        
        根据论文4.4节，这是解决WAR冲突的关键函数：
        1. 查询内存利用率
        2. 获取块进行KV cache存储
        3. 更新内存利用率
        
        整个过程必须是原子性的。
        """
        start_time = time.time()
        
        # 原子性分配 - 解决WAR冲突的核心
        with self.allocation_lock:
            try:
                # 第一步：查询内存利用率
                utilization = self.memory_utilization.query_utilization()
                
                if utilization["free_blocks"] < num_blocks:
                    logger.warning(
                        f"Insufficient free blocks for allocation: "
                        f"requested={num_blocks}, available={utilization['free_blocks']}"
                    )
                    self.stats["allocation_failures"] += 1
                    return None
                
                # 第二步：获取块进行KV cache存储
                allocated_block_ids = []
                
                # 从空闲块中分配
                free_blocks_list = list(self.free_block_ids)
                if len(free_blocks_list) < num_blocks:
                    logger.error("Inconsistent free block count")
                    return None
                    
                for i in range(num_blocks):
                    block_id = free_blocks_list[i]
                    allocated_block_ids.append(block_id)
                    
                    # 更新块状态
                    self.blocks[block_id].state = BlockState.ALLOCATED
                    self.blocks[block_id].request_id = request_id
                    self.blocks[block_id].layer_id = layer_id
                    self.blocks[block_id].allocated_time = time.time()
                    self.blocks[block_id].allocator_role = requester_role
                    
                    # 从空闲集合中移除
                    self.free_block_ids.remove(block_id)
                
                # 第三步：原子性更新内存利用率
                success = self.memory_utilization.atomic_allocate(num_blocks)
                if not success:
                    # 回滚分配
                    self._rollback_allocation(allocated_block_ids)
                    logger.error("Failed to update memory utilization atomically")
                    return None
                
                # 更新分配记录
                if request_id not in self.allocated_blocks:
                    self.allocated_blocks[request_id] = []
                self.allocated_blocks[request_id].extend(allocated_block_ids)
                
                # 创建页表项
                self._create_page_table_entries(request_id, layer_id, allocated_block_ids)
                
                # 更新统计信息
                self.stats["total_allocations"] += 1
                allocation_time = time.time() - start_time
                self.stats["avg_allocation_time"] = (
                    (self.stats["avg_allocation_time"] * (self.stats["total_allocations"] - 1) + allocation_time) /
                    self.stats["total_allocations"]
                )
                
                current_utilization = utilization["allocated_blocks"] / utilization["total_blocks"]
                self.stats["peak_utilization"] = max(self.stats["peak_utilization"], current_utilization)
                
                logger.debug(
                    f"Allocated {num_blocks} blocks for request {request_id} layer {layer_id} "
                    f"by {requester_role.name}: {allocated_block_ids}"
                )
                
                return allocated_block_ids
                
            except Exception as e:
                logger.error(f"Error during KV cache allocation: {e}")
                self.stats["allocation_failures"] += 1
                return None
                
    def _create_page_table_entries(
        self, 
        request_id: str, 
        layer_id: int, 
        block_ids: List[int]
    ):
        """创建页表项（vLLM风格的paged storage）"""
        with self.page_table_lock:
            if request_id not in self.page_tables:
                self.page_tables[request_id] = {}
                
            if layer_id not in self.page_tables[request_id]:
                self.page_tables[request_id][layer_id] = []
                
            # 为每个块创建页表项
            for i, block_id in enumerate(block_ids):
                entry = PageTableEntry(
                    virtual_page_id=i,
                    physical_block_id=block_id,
                    request_id=request_id,
                    layer_id=layer_id,
                )
                self.page_tables[request_id][layer_id].append(entry)
                
    def get_block_table_index(
        self, 
        request_id: str, 
        layer_id: int
    ) -> Optional[List[int]]:
        """
        获取块表索引（vLLM风格）
        
        根据论文4.4节："Once the block table index is determined, 
        the access of the KV cache can be conducted without conflicts."
        """
        with self.page_table_lock:
            if (request_id in self.page_tables and 
                layer_id in self.page_tables[request_id]):
                
                entries = self.page_tables[request_id][layer_id]
                return [entry.physical_block_id for entry in entries if entry.is_valid]
            else:
                return None
                
    def access_kv_cache(
        self,
        request_id: str,
        layer_id: int,
        accessor_role: InstanceRole,
    ) -> Optional[List[torch.Tensor]]:
        """
        访问KV cache（通过块表索引）
        
        根据论文："Once the block table index is determined, 
        the access of the KV cache can be conducted without conflicts."
        """
        with self.access_lock:
            # 获取块表索引
            block_indices = self.get_block_table_index(request_id, layer_id)
            
            if block_indices is None:
                logger.warning(f"No block table found for request {request_id} layer {layer_id}")
                return None
                
            # 通过块表索引访问KV cache
            kv_tensors = []
            for block_id in block_indices:
                if block_id in self.blocks:
                    block_info = self.blocks[block_id]
                    
                    # 更新访问时间
                    block_info.last_access_time = time.time()
                    
                    # 这里应该返回实际的KV cache tensor
                    # 简化实现，返回占位符tensor
                    kv_tensor = torch.zeros(
                        (self.page_size, 64, 128),  # 示例维度
                        dtype=self.dtype,
                        device=self.device
                    )
                    kv_tensors.append(kv_tensor)
                else:
                    logger.error(f"Block {block_id} not found")
                    return None
                    
            return kv_tensors
            
    def deallocate_kv_cache_blocks(self, request_id: str) -> bool:
        """释放KV cache块（原子性操作）"""
        with self.allocation_lock:
            try:
                if request_id not in self.allocated_blocks:
                    logger.warning(f"No allocated blocks found for request {request_id}")
                    return True  # 已经释放或从未分配
                    
                block_ids = self.allocated_blocks[request_id]
                num_blocks = len(block_ids)
                
                # 释放块
                for block_id in block_ids:
                    if block_id in self.blocks:
                        self.blocks[block_id].state = BlockState.FREE
                        self.blocks[block_id].request_id = None
                        self.blocks[block_id].layer_id = None
                        self.blocks[block_id].allocator_role = None
                        
                        # 添加回空闲集合
                        self.free_block_ids.add(block_id)
                        
                # 原子性更新内存利用率
                success = self.memory_utilization.atomic_deallocate(num_blocks)
                if not success:
                    logger.error(f"Failed to update memory utilization during deallocation")
                    
                # 清理分配记录
                del self.allocated_blocks[request_id]
                
                # 清理页表
                self._cleanup_page_table(request_id)
                
                # 更新统计信息
                self.stats["total_deallocations"] += 1
                
                logger.debug(f"Deallocated {num_blocks} blocks for request {request_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error during KV cache deallocation: {e}")
                return False
                
    def _cleanup_page_table(self, request_id: str):
        """清理页表"""
        with self.page_table_lock:
            if request_id in self.page_tables:
                del self.page_tables[request_id]
                
    def _rollback_allocation(self, block_ids: List[int]):
        """回滚分配操作"""
        for block_id in block_ids:
            if block_id in self.blocks:
                self.blocks[block_id].state = BlockState.FREE
                self.blocks[block_id].request_id = None
                self.blocks[block_id].layer_id = None
                self.blocks[block_id].allocator_role = None
                self.free_block_ids.add(block_id)
                
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        utilization = self.memory_utilization.query_utilization()
        
        return {
            "total_blocks": self.total_blocks,
            "allocated_blocks": utilization["allocated_blocks"],
            "free_blocks": utilization["free_blocks"],
            "reserved_blocks": utilization["reserved_blocks"],
            "utilization_ratio": utilization["utilization_ratio"],
            "free_ratio": utilization["free_ratio"],
            "block_size": self.block_size,
            "total_memory_bytes": self.total_blocks * self.block_size,
            "allocated_memory_bytes": utilization["allocated_blocks"] * self.block_size,
            "free_memory_bytes": utilization["free_blocks"] * self.block_size,
        }
        
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "memory_usage": self.get_memory_usage(),
            "active_requests": len(self.allocated_blocks),
            "total_page_tables": len(self.page_tables),
            "model_weights_count": len(self.model_weights),
            "weight_access_stats": dict(self.weight_access_count),
        }
        
    def detect_war_conflicts(self) -> List[Dict]:
        """检测潜在的WAR冲突"""
        conflicts = []
        
        # 检查同时访问相同块的不同角色
        block_access_map: Dict[int, List[InstanceRole]] = {}
        
        for request_id, block_ids in self.allocated_blocks.items():
            for block_id in block_ids:
                if block_id in self.blocks:
                    role = self.blocks[block_id].allocator_role
                    if role:
                        if block_id not in block_access_map:
                            block_access_map[block_id] = []
                        block_access_map[block_id].append(role)
                        
        # 查找冲突
        for block_id, roles in block_access_map.items():
            unique_roles = set(roles)
            if len(unique_roles) > 1 and InstanceRole.PREFILL in unique_roles and InstanceRole.DECODE in unique_roles:
                conflicts.append({
                    "block_id": block_id,
                    "conflicting_roles": list(unique_roles),
                    "conflict_type": "WAR",
                    "description": "Prefill and decode workers accessing same block"
                })
                
        return conflicts
        
    def force_garbage_collection(self) -> int:
        """强制垃圾回收"""
        collected_blocks = 0
        current_time = time.time()
        
        with self.allocation_lock:
            # 查找长时间未访问的块
            expired_requests = []
            
            for request_id, block_ids in self.allocated_blocks.items():
                all_expired = True
                for block_id in block_ids:
                    if block_id in self.blocks:
                        block_info = self.blocks[block_id]
                        # 如果块超过1小时未访问，认为过期
                        if current_time - block_info.last_access_time < 3600:
                            all_expired = False
                            break
                            
                if all_expired:
                    expired_requests.append(request_id)
                    
            # 释放过期请求的块
            for request_id in expired_requests:
                if self.deallocate_kv_cache_blocks(request_id):
                    collected_blocks += len(self.allocated_blocks.get(request_id, []))
                    
        logger.info(f"Garbage collection completed: {collected_blocks} blocks collected")
        return collected_blocks
        
    def get_status(self) -> Dict:
        """获取管理器状态"""
        return {
            "memory_usage": self.get_memory_usage(),
            "statistics": self.get_statistics(),
            "war_conflicts": self.detect_war_conflicts(),
            "configuration": {
                "total_blocks": self.total_blocks,
                "block_size": self.block_size,
                "page_size": self.page_size,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "enable_prefix_caching": self.enable_prefix_caching,
            },
        }