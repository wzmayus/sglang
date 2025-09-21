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
"""Unified storage manager for Semi-PD KV cache management."""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.semi_pd.utils import InstanceRole
from sglang.srt.mem_cache.chunk_cache import ChunkCache

logger = logging.getLogger(__name__)


class BlockState(Enum):
    FREE = "free"           # 空闲块
    ALLOCATED = "allocated" # 已分配但未使用
    IN_USE = "in_use"      # 正在使用中
    RESERVED = "reserved"   # 预留块


@dataclass
class BlockInfo:
    """KV cache块信息"""
    block_id: int
    state: BlockState
    request_id: Optional[str] = None
    allocated_time: Optional[float] = None
    last_access_time: Optional[float] = None
    reference_count: int = 0
    size: int = 0
    
    
@dataclass
class AllocationRequest:
    """分配请求"""
    request_id: str
    num_blocks: int
    priority: int = 0
    requester_role: InstanceRole = InstanceRole.OTHER
    

class UnifiedStorageManager:
    """
    统一存储管理器
    
    负责管理Semi-PD架构中的KV cache存储，确保：
    1. 原子性：块分配操作的原子性
    2. 一致性：P和D实例之间的读写一致性
    3. 并发安全：多进程访问的线程安全
    4. 内存效率：避免内存碎片和泄漏
    """
    
    def __init__(
        self,
        total_blocks: int,
        block_size: int,
        device: torch.device,
        enable_prefix_caching: bool = False,
    ):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.device = device
        self.enable_prefix_caching = enable_prefix_caching
        
        # 块管理
        self.blocks: Dict[int, BlockInfo] = {}
        self.free_blocks: Set[int] = set()
        self.allocated_blocks: Dict[str, List[int]] = {}  # request_id -> block_ids
        
        # 同步控制
        self.allocation_lock = threading.RLock()
        self.access_lock = threading.RLock()
        
        # 统计信息
        self.allocation_stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "peak_usage": 0,
            "current_usage": 0,
            "allocation_failures": 0,
        }
        
        # 初始化块池
        self._initialize_blocks()
        
        logger.info(
            f"Initialized unified storage manager: "
            f"{total_blocks} blocks, {block_size} bytes per block"
        )
        
    def _initialize_blocks(self):
        """初始化块池"""
        for block_id in range(self.total_blocks):
            block_info = BlockInfo(
                block_id=block_id,
                state=BlockState.FREE,
            )
            self.blocks[block_id] = block_info
            self.free_blocks.add(block_id)
            
    def allocate_blocks(
        self, 
        request_id: str, 
        num_blocks: int,
        requester_role: InstanceRole = InstanceRole.OTHER,
        priority: int = 0,
    ) -> Optional[List[int]]:
        """
        原子性地分配KV cache块
        
        Args:
            request_id: 请求ID
            num_blocks: 需要的块数量
            requester_role: 请求者角色（PREFILL或DECODE）
            priority: 优先级（数值越大优先级越高）
            
        Returns:
            分配的块ID列表，失败时返回None
        """
        with self.allocation_lock:
            try:
                # 检查是否有足够的空闲块
                if len(self.free_blocks) < num_blocks:
                    logger.warning(
                        f"Insufficient free blocks for request {request_id}: "
                        f"requested {num_blocks}, available {len(self.free_blocks)}"
                    )
                    
                    # 尝试回收一些块
                    if self._try_reclaim_blocks(num_blocks):
                        logger.info(f"Reclaimed blocks for request {request_id}")
                    else:
                        self.allocation_stats["allocation_failures"] += 1
                        return None
                        
                # 分配块
                allocated_block_ids = []
                current_time = time.time()
                
                for _ in range(num_blocks):
                    if not self.free_blocks:
                        # 回滚已分配的块
                        self._rollback_allocation(allocated_block_ids)
                        self.allocation_stats["allocation_failures"] += 1
                        return None
                        
                    block_id = self.free_blocks.pop()
                    block_info = self.blocks[block_id]
                    
                    # 更新块状态
                    block_info.state = BlockState.ALLOCATED
                    block_info.request_id = request_id
                    block_info.allocated_time = current_time
                    block_info.last_access_time = current_time
                    block_info.reference_count = 1
                    
                    allocated_block_ids.append(block_id)
                    
                # 记录分配信息
                if request_id in self.allocated_blocks:
                    self.allocated_blocks[request_id].extend(allocated_block_ids)
                else:
                    self.allocated_blocks[request_id] = allocated_block_ids
                    
                # 更新统计信息
                self.allocation_stats["total_allocations"] += 1
                self.allocation_stats["current_usage"] += num_blocks
                self.allocation_stats["peak_usage"] = max(
                    self.allocation_stats["peak_usage"],
                    self.allocation_stats["current_usage"]
                )
                
                logger.debug(
                    f"Allocated {num_blocks} blocks for request {request_id}: {allocated_block_ids}"
                )
                
                return allocated_block_ids
                
            except Exception as e:
                logger.error(f"Error allocating blocks for request {request_id}: {e}")
                self.allocation_stats["allocation_failures"] += 1
                return None
                
    def deallocate_blocks(self, request_id: str) -> bool:
        """
        释放请求的所有块
        
        Args:
            request_id: 请求ID
            
        Returns:
            是否成功释放
        """
        with self.allocation_lock:
            try:
                if request_id not in self.allocated_blocks:
                    logger.warning(f"Request {request_id} has no allocated blocks")
                    return False
                    
                block_ids = self.allocated_blocks[request_id]
                
                # 释放所有块
                for block_id in block_ids:
                    block_info = self.blocks[block_id]
                    
                    # 减少引用计数
                    block_info.reference_count -= 1
                    
                    # 如果引用计数为0，释放块
                    if block_info.reference_count <= 0:
                        block_info.state = BlockState.FREE
                        block_info.request_id = None
                        block_info.allocated_time = None
                        block_info.last_access_time = None
                        block_info.reference_count = 0
                        
                        self.free_blocks.add(block_id)
                        
                # 清除分配记录
                del self.allocated_blocks[request_id]
                
                # 更新统计信息
                self.allocation_stats["total_deallocations"] += 1
                self.allocation_stats["current_usage"] -= len(block_ids)
                
                logger.debug(f"Deallocated {len(block_ids)} blocks for request {request_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error deallocating blocks for request {request_id}: {e}")
                return False
                
    def get_block_handles(self, request_id: str) -> Optional[List[torch.Tensor]]:
        """
        获取请求的块句柄，用于跨进程共享
        
        Args:
            request_id: 请求ID
            
        Returns:
            块句柄列表
        """
        with self.access_lock:
            try:
                if request_id not in self.allocated_blocks:
                    return None
                    
                block_ids = self.allocated_blocks[request_id]
                handles = []
                
                for block_id in block_ids:
                    # 这里应该返回实际的tensor句柄
                    # 简化实现，返回块ID
                    handles.append(torch.tensor([block_id], device=self.device))
                    
                    # 更新访问时间
                    self.blocks[block_id].last_access_time = time.time()
                    
                return handles
                
            except Exception as e:
                logger.error(f"Error getting block handles for request {request_id}: {e}")
                return None
                
    def share_blocks(self, source_request_id: str, target_request_id: str) -> bool:
        """
        在请求之间共享块（用于prefix caching）
        
        Args:
            source_request_id: 源请求ID
            target_request_id: 目标请求ID
            
        Returns:
            是否成功共享
        """
        if not self.enable_prefix_caching:
            return False
            
        with self.allocation_lock:
            try:
                if source_request_id not in self.allocated_blocks:
                    return False
                    
                source_blocks = self.allocated_blocks[source_request_id]
                
                # 增加引用计数
                for block_id in source_blocks:
                    self.blocks[block_id].reference_count += 1
                    
                # 添加到目标请求
                if target_request_id in self.allocated_blocks:
                    self.allocated_blocks[target_request_id].extend(source_blocks)
                else:
                    self.allocated_blocks[target_request_id] = source_blocks.copy()
                    
                logger.debug(
                    f"Shared {len(source_blocks)} blocks from {source_request_id} to {target_request_id}"
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Error sharing blocks: {e}")
                return False
                
    def _try_reclaim_blocks(self, needed_blocks: int) -> bool:
        """
        尝试回收块以满足分配需求
        
        Args:
            needed_blocks: 需要的块数量
            
        Returns:
            是否成功回收足够的块
        """
        # 查找可回收的块（例如长时间未访问的块）
        current_time = time.time()
        reclaim_threshold = 300.0  # 5分钟未访问
        
        reclaimable_blocks = []
        
        for block_id, block_info in self.blocks.items():
            if (block_info.state == BlockState.ALLOCATED and 
                block_info.last_access_time and
                current_time - block_info.last_access_time > reclaim_threshold and
                block_info.reference_count <= 1):
                reclaimable_blocks.append(block_id)
                
        if len(reclaimable_blocks) >= needed_blocks:
            # 回收块
            for block_id in reclaimable_blocks[:needed_blocks]:
                block_info = self.blocks[block_id]
                
                # 从原请求中移除
                if block_info.request_id in self.allocated_blocks:
                    try:
                        self.allocated_blocks[block_info.request_id].remove(block_id)
                        if not self.allocated_blocks[block_info.request_id]:
                            del self.allocated_blocks[block_info.request_id]
                    except ValueError:
                        pass
                        
                # 重置块状态
                block_info.state = BlockState.FREE
                block_info.request_id = None
                block_info.allocated_time = None
                block_info.last_access_time = None
                block_info.reference_count = 0
                
                self.free_blocks.add(block_id)
                
            logger.info(f"Reclaimed {len(reclaimable_blocks[:needed_blocks])} blocks")
            return True
            
        return False
        
    def _rollback_allocation(self, allocated_block_ids: List[int]):
        """回滚分配操作"""
        for block_id in allocated_block_ids:
            block_info = self.blocks[block_id]
            block_info.state = BlockState.FREE
            block_info.request_id = None
            block_info.allocated_time = None
            block_info.last_access_time = None
            block_info.reference_count = 0
            
            self.free_blocks.add(block_id)
            
    def get_memory_usage(self) -> Dict:
        """获取内存使用统计"""
        with self.allocation_lock:
            free_blocks = len(self.free_blocks)
            allocated_blocks = self.total_blocks - free_blocks
            
            return {
                "total_blocks": self.total_blocks,
                "free_blocks": free_blocks,
                "allocated_blocks": allocated_blocks,
                "utilization": allocated_blocks / self.total_blocks,
                "block_size": self.block_size,
                "total_memory": self.total_blocks * self.block_size,
                "free_memory": free_blocks * self.block_size,
                "allocated_memory": allocated_blocks * self.block_size,
                "stats": self.allocation_stats.copy(),
            }
            
    def cleanup_expired_blocks(self, max_age_seconds: float = 3600.0):
        """清理过期的块"""
        current_time = time.time()
        expired_requests = []
        
        with self.allocation_lock:
            for request_id, block_ids in self.allocated_blocks.items():
                # 检查是否有块过期
                has_expired = False
                for block_id in block_ids:
                    block_info = self.blocks[block_id]
                    if (block_info.allocated_time and 
                        current_time - block_info.allocated_time > max_age_seconds):
                        has_expired = True
                        break
                        
                if has_expired:
                    expired_requests.append(request_id)
                    
            # 清理过期请求
            for request_id in expired_requests:
                logger.info(f"Cleaning up expired request: {request_id}")
                self.deallocate_blocks(request_id)
                
    def force_gc(self):
        """强制垃圾回收"""
        logger.info("Performing forced garbage collection")
        
        with self.allocation_lock:
            # 清理引用计数为0的块
            orphaned_blocks = []
            
            for block_id, block_info in self.blocks.items():
                if (block_info.state != BlockState.FREE and 
                    block_info.reference_count <= 0):
                    orphaned_blocks.append(block_id)
                    
            for block_id in orphaned_blocks:
                block_info = self.blocks[block_id]
                
                # 从分配记录中移除
                if block_info.request_id in self.allocated_blocks:
                    try:
                        self.allocated_blocks[block_info.request_id].remove(block_id)
                        if not self.allocated_blocks[block_info.request_id]:
                            del self.allocated_blocks[block_info.request_id]
                    except ValueError:
                        pass
                        
                # 重置块状态
                block_info.state = BlockState.FREE
                block_info.request_id = None
                block_info.allocated_time = None
                block_info.last_access_time = None
                block_info.reference_count = 0
                
                self.free_blocks.add(block_id)
                
            if orphaned_blocks:
                logger.info(f"Garbage collected {len(orphaned_blocks)} orphaned blocks")
                
    def get_status(self) -> Dict:
        """获取管理器状态"""
        memory_usage = self.get_memory_usage()
        
        return {
            "memory_usage": memory_usage,
            "active_requests": len(self.allocated_blocks),
            "enable_prefix_caching": self.enable_prefix_caching,
            "device": str(self.device),
        }