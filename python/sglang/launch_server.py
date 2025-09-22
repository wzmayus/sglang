"""Launch the inference server with Semi-PD support."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    
    # 显示Semi-PD配置信息
    if getattr(server_args, 'enable_semi_pd_coordinator', False) or getattr(server_args, 'enable_semi_pd', False):
        print("=== Semi-PD Configuration ===")
        print(f"Semi-PD Coordinator: {'✓' if getattr(server_args, 'enable_semi_pd_coordinator', False) else '✗'}")
        print(f"Unified Memory Manager: {'✓' if getattr(server_args, 'enable_unified_memory', False) else '✗'}")
        print(f"SLO-aware Algorithm: {'✓' if getattr(server_args, 'enable_slo_aware', False) else '✗'}")
        
        if getattr(server_args, 'enable_unified_memory', False):
            print(f"  - Memory Blocks: {getattr(server_args, 'unified_memory_blocks', 1000)}")
            print(f"  - Block Size: {getattr(server_args, 'unified_memory_block_size', 4096)} bytes")
            
        if getattr(server_args, 'enable_slo_aware', False):
            print(f"  - TTFT Target: {getattr(server_args, 'slo_ttft_target', 100.0)} ms")
            print(f"  - TPOT Target: {getattr(server_args, 'slo_tpot_target', 50.0)} ms")
            
        print(f"  - Initial SM Allocation: P{getattr(server_args, 'initial_prefill_sm', 70)}% D{getattr(server_args, 'initial_decode_sm', 30)}%")
        print()

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
