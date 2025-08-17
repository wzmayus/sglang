import logging
import os
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)


from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

hiradix_core = load(
    name="hiradix_core",
    sources=[f"{_abs_path}/cpp_hiradix_tree/hiradix_core.cpp"],
    extra_cflags=["-O3", "-std=c++17"],
)


class HiRadixCacheCpp(HiRadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        hicache_io_backend: str,
        hicache_mem_layout: str,
        hicache_storage_backend: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = "best_effort",
    ):
        self.device = torch.device(token_to_kv_pool_allocator.device)
        self.hiradix_core = hiradix_core.HiRadixTreeCore(page_size, self.device)
        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tp_cache_group=tp_cache_group,
            page_size=page_size,
            hicache_ratio=hicache_ratio,
            hicache_size=hicache_size,
            hicache_write_policy=hicache_write_policy,
            hicache_io_backend=hicache_io_backend,
            hicache_mem_layout=hicache_mem_layout,
            hicache_storage_backend=hicache_storage_backend,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
        )

    def reset(self):
        self.hiradix_core.reset()
        self.root_node = self.hiradix_core.root_node
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()

    def match_prefix(self, key: List[int], **kwargs):
        device_indices, last_device_node, last_host_node, host_hit_length = (
            self.hiradix_core.match_prefix(key)
        )
        return MatchResult(
            device_indices=device_indices,
            last_device_node=last_device_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def _insert_helper(self, node: TreeNode, key: List[int], value=None):
        new_node, matched_len = self.hiradix_core.insert_helper(node, key, value)

        if new_node is not None:
            # write through every new nodes
            self.write_backup(new_node)
            if self.enable_storage:
                last_hash = new_node.parent.get_last_hash_value()
                hash_value = []
                for idx in range(0, len(key), self.page_size):
                    hash_value.append(
                        self.cache_controller.get_hash_str(
                            key[idx : idx + self.page_size],
                            prior_hash=last_hash,
                        )
                    )
                    last_hash = hash_value[-1]
                new_node.hash_value = hash_value
        return matched_len

    def _insert_helper_host(
        self, node: TreeNode, key: List[int], host_value, hash_value
    ):
        return self.hiradix_core.insert_helper_host(node, key, host_value, hash_value)

    def evict(self, num_tokens: int):
        to_free = self.hiradix_core.evict_device(num_tokens)
        if to_free.numel() > 0:
            self.cache_controller.mem_pool_device_allocator.free(to_free)

    def evict_host(self, num_tokens: int):
        to_free = self.hiradix_core.evict_host(num_tokens)
        if to_free.numel() > 0:
            self.cache_controller.evict_host(to_free)

    def _split_node(self, key, child: TreeNode, split_len: int):
        return self.hiradix_core.split_node(child, split_len)

    def inc_lock_ref(self, node: TreeNode):
        return self.hiradix_core.inc_lock_ref(node)

    def dec_lock_ref(self, node: TreeNode):
        return self.hiradix_core.dec_lock_ref(node)

    def evictable_size(self):
        return self.hiradix_core.evictable_size()

    def protected_size(self):
        return self.hiradix_core.protected_size()

    def assign_value(
        self,
        node: TreeNode,
        new_indices: torch.Tensor,
    ):
        self.hiradix_core.set_node_value(
            node,
            new_indices,
        )

    def insert_pending_request(
        self,
        node: TreeNode,
        key: List[int],
    ):
        self.hiradix_core.insert_pending_request(node, key)

    def scheduling(self, requests):
        return self.hiradix_core.scheduling(
            requests,
        )
