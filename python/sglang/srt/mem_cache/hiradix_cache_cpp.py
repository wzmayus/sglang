import logging
import os
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import HiRadixCache, TreeNode

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
        self.device = self.token_to_kv_pool_allocator.device
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
        return self.hiradix_core.insert_helper(node, key, value)

    def _insert_helper_host(
        self, node: TreeNode, key: List[int], host_value, hash_value
    ):
        return self.hiradix_core.insert_helper_host(node, key, host_value, hash_value)

    def evict(self, num_tokens: int):
        to_free = self.hiradix_core.evict_device(num_tokens)
        if to_free and to_free.numel() > 0:
            self.cache_controller.mem_pool_device_allocator.free(to_free)

    def evict_host(self, num_tokens: int):
        to_free = self.hiradix_core.evict_host(num_tokens)
        if to_free and to_free.numel() > 0:
            self.cache_controller.mem_pool_host_allocator.free(to_free)

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

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in nodes_to_load:
            self.hiradix_core.set_device_indices(
                node, device_indices[offset : offset + len(node.host_value)]
            )
            offset += len(node.host_value)
        self.inc_lock_ref(last_hit_node)

        return device_indices
