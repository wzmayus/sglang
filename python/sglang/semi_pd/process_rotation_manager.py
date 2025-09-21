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
"""Process rotation manager for Semi-PD dynamic SM allocation."""

import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import zmq

from sglang.semi_pd.utils import InstanceRole
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    ACTIVE = "active"      # 当前正在推理的进程
    STANDBY = "standby"    # 休眠进程，准备接管
    SWITCHING = "switching"  # 正在切换中
    TERMINATED = "terminated"  # 已终止


@dataclass
class SMAllocation:
    """SM资源分配配置"""
    prefill_percentage: int  # Prefill进程的SM百分比
    decode_percentage: int   # Decode进程的SM百分比
    
    def __post_init__(self):
        if self.prefill_percentage + self.decode_percentage > 100:
            raise ValueError("Total SM allocation cannot exceed 100%")


@dataclass
class ProcessInfo:
    """进程信息"""
    process: mp.Process
    state: ProcessState
    sm_allocation: SMAllocation
    creation_time: float
    last_switch_time: Optional[float] = None


class ProcessRotationManager:
    """
    进程轮换管理器
    
    实现两组进程轮换机制：
    1. 推理进程组：当前正在处理请求的进程
    2. 休眠进程组：预先启动的备用进程
    
    当需要调整SM配比时，休眠进程会用新配比重新启动，
    然后在推理进程完成当前step后进行角色切换。
    """
    
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        initial_sm_allocation: SMAllocation,
    ):
        self.server_args = server_args
        self.port_args = port_args
        self.current_sm_allocation = initial_sm_allocation
        
        # 进程组管理
        self.active_processes: Dict[InstanceRole, ProcessInfo] = {}
        self.standby_processes: Dict[InstanceRole, ProcessInfo] = {}
        
        # 同步控制
        self.switch_lock = threading.Lock()
        self.switch_event = threading.Event()
        self.switch_requested = False
        self.target_sm_allocation: Optional[SMAllocation] = None
        
        # ZMQ通信
        self.context = zmq.Context()
        self.control_socket = self.context.socket(zmq.PUB)
        self.control_socket.bind("ipc:///tmp/semi_pd_control")
        
        # 监控线程
        self.monitor_thread = None
        self.running = False
        
    def start(self):
        """启动进程轮换管理器"""
        logger.info("Starting process rotation manager")
        self.running = True
        
        # 启动初始进程组
        self._start_initial_processes()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_processes)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """停止进程轮换管理器"""
        logger.info("Stopping process rotation manager")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        # 终止所有进程
        self._terminate_all_processes()
        self.context.term()
        
    def request_sm_reallocation(self, new_allocation: SMAllocation) -> bool:
        """
        请求SM重新分配
        
        Args:
            new_allocation: 新的SM分配配置
            
        Returns:
            bool: 是否成功启动重新分配流程
        """
        with self.switch_lock:
            if self.switch_requested:
                logger.warning("SM reallocation already in progress")
                return False
                
            logger.info(
                f"Requesting SM reallocation: "
                f"P:{new_allocation.prefill_percentage}% "
                f"D:{new_allocation.decode_percentage}%"
            )
            
            self.target_sm_allocation = new_allocation
            self.switch_requested = True
            
            # 启动新的休眠进程组
            success = self._start_standby_processes(new_allocation)
            if success:
                # 通知活跃进程准备切换
                self._notify_active_processes_prepare_switch()
                
            return success
            
    def _start_initial_processes(self):
        """启动初始进程组"""
        logger.info("Starting initial process group")
        
        for role in [InstanceRole.PREFILL, InstanceRole.DECODE]:
            process_info = self._create_process(role, self.current_sm_allocation)
            if process_info:
                self.active_processes[role] = process_info
                logger.info(f"Started initial {role.name} process")
            else:
                raise RuntimeError(f"Failed to start initial {role.name} process")
                
    def _start_standby_processes(self, sm_allocation: SMAllocation) -> bool:
        """启动休眠进程组"""
        logger.info("Starting standby process group")
        
        try:
            for role in [InstanceRole.PREFILL, InstanceRole.DECODE]:
                # 如果已有休眠进程，先终止
                if role in self.standby_processes:
                    self._terminate_process(self.standby_processes[role])
                    del self.standby_processes[role]
                    
                # 创建新的休眠进程
                process_info = self._create_process(role, sm_allocation)
                if process_info:
                    process_info.state = ProcessState.STANDBY
                    self.standby_processes[role] = process_info
                    logger.info(f"Started standby {role.name} process")
                else:
                    logger.error(f"Failed to start standby {role.name} process")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error starting standby processes: {e}")
            return False
            
    def _create_process(
        self, 
        role: InstanceRole, 
        sm_allocation: SMAllocation
    ) -> Optional[ProcessInfo]:
        """创建新进程"""
        try:
            from sglang.srt.managers.semi_pd_scheduler import run_scheduler_process
            
            # 设置SM分配环境变量
            if role == InstanceRole.PREFILL:
                sm_percentage = sm_allocation.prefill_percentage
            else:
                sm_percentage = sm_allocation.decode_percentage
                
            # 创建进程
            reader, writer = mp.Pipe(duplex=False)
            
            process = mp.Process(
                target=self._process_wrapper,
                args=(
                    run_scheduler_process,
                    self.server_args,
                    self.port_args,
                    0,  # gpu_id
                    0,  # tp_rank
                    0,  # moe_ep_rank
                    0,  # pp_rank
                    None,  # dp_rank
                    writer,
                    None,  # ipc_info_queue
                    False,  # bypass_load_weight
                    role,
                    sm_percentage,
                ),
            )
            
            process.start()
            
            # 等待进程就绪
            try:
                data = reader.recv()
                if data.get("status") == "ready":
                    return ProcessInfo(
                        process=process,
                        state=ProcessState.ACTIVE,
                        sm_allocation=sm_allocation,
                        creation_time=time.time(),
                    )
                else:
                    logger.error(f"Process failed to start: {data}")
                    process.terminate()
                    return None
                    
            except Exception as e:
                logger.error(f"Error waiting for process ready: {e}")
                process.terminate()
                return None
                
        except Exception as e:
            logger.error(f"Error creating process: {e}")
            return None
            
    def _process_wrapper(
        self,
        target_func,
        server_args,
        port_args,
        gpu_id,
        tp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        pipe_writer,
        ipc_info_queue,
        bypass_load_weight,
        instance_role,
        sm_percentage,
    ):
        """进程包装器，设置SM分配环境变量"""
        # 设置SM百分比
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(sm_percentage)
        
        # 调用原始函数
        target_func(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            pipe_writer,
            ipc_info_queue,
            bypass_load_weight,
            instance_role,
        )
        
    def _notify_active_processes_prepare_switch(self):
        """通知活跃进程准备切换"""
        logger.info("Notifying active processes to prepare for switch")
        
        switch_signal = {
            "action": "prepare_switch",
            "timestamp": time.time(),
        }
        
        self.control_socket.send_pyobj(switch_signal)
        
    def _perform_switch(self):
        """执行进程切换"""
        logger.info("Performing process switch")
        
        try:
            # 等待当前step完成的信号
            # 这里应该从活跃进程接收完成信号
            
            # 执行切换
            old_active = self.active_processes.copy()
            
            # 将休眠进程提升为活跃进程
            for role, standby_info in self.standby_processes.items():
                standby_info.state = ProcessState.ACTIVE
                standby_info.last_switch_time = time.time()
                self.active_processes[role] = standby_info
                
            # 终止旧的活跃进程
            for role, old_info in old_active.items():
                self._terminate_process(old_info)
                
            # 清空休眠进程
            self.standby_processes.clear()
            
            # 更新当前SM分配
            self.current_sm_allocation = self.target_sm_allocation
            
            # 重置切换状态
            self.switch_requested = False
            self.target_sm_allocation = None
            self.switch_event.set()
            
            logger.info("Process switch completed successfully")
            
        except Exception as e:
            logger.error(f"Error during process switch: {e}")
            # 回滚操作
            self._rollback_switch()
            
    def _rollback_switch(self):
        """回滚切换操作"""
        logger.warning("Rolling back process switch")
        
        # 终止失败的休眠进程
        for role, standby_info in self.standby_processes.items():
            self._terminate_process(standby_info)
            
        self.standby_processes.clear()
        self.switch_requested = False
        self.target_sm_allocation = None
        
    def _monitor_processes(self):
        """监控进程状态"""
        while self.running:
            try:
                # 检查活跃进程健康状态
                for role, process_info in list(self.active_processes.items()):
                    if not process_info.process.is_alive():
                        logger.error(f"Active {role.name} process died unexpectedly")
                        # 这里可以实现自动重启逻辑
                        
                # 检查是否需要执行切换
                if self.switch_requested and self._all_standby_processes_ready():
                    # 检查是否收到切换就绪信号
                    if self._check_switch_ready_signal():
                        self._perform_switch()
                        
                time.sleep(1.0)  # 监控间隔
                
            except Exception as e:
                logger.error(f"Error in process monitor: {e}")
                
    def _all_standby_processes_ready(self) -> bool:
        """检查所有休眠进程是否就绪"""
        required_roles = {InstanceRole.PREFILL, InstanceRole.DECODE}
        standby_roles = set(self.standby_processes.keys())
        
        if not required_roles.issubset(standby_roles):
            return False
            
        for process_info in self.standby_processes.values():
            if not process_info.process.is_alive():
                return False
                
        return True
        
    def _check_switch_ready_signal(self) -> bool:
        """检查是否收到切换就绪信号"""
        # 这里应该检查来自活跃进程的就绪信号
        # 简化实现，假设在一定时间后就绪
        return True
        
    def _terminate_process(self, process_info: ProcessInfo):
        """终止进程"""
        try:
            if process_info.process.is_alive():
                process_info.process.terminate()
                process_info.process.join(timeout=5.0)
                
                if process_info.process.is_alive():
                    logger.warning("Process did not terminate gracefully, killing")
                    process_info.process.kill()
                    
            process_info.state = ProcessState.TERMINATED
            
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
            
    def _terminate_all_processes(self):
        """终止所有进程"""
        all_processes = list(self.active_processes.values()) + list(self.standby_processes.values())
        
        for process_info in all_processes:
            self._terminate_process(process_info)
            
        self.active_processes.clear()
        self.standby_processes.clear()
        
    def get_status(self) -> Dict:
        """获取管理器状态"""
        return {
            "current_sm_allocation": {
                "prefill_percentage": self.current_sm_allocation.prefill_percentage,
                "decode_percentage": self.current_sm_allocation.decode_percentage,
            },
            "switch_requested": self.switch_requested,
            "target_sm_allocation": {
                "prefill_percentage": self.target_sm_allocation.prefill_percentage,
                "decode_percentage": self.target_sm_allocation.decode_percentage,
            } if self.target_sm_allocation else None,
            "active_processes": {
                role.name: {
                    "pid": info.process.pid,
                    "state": info.state.value,
                    "creation_time": info.creation_time,
                    "last_switch_time": info.last_switch_time,
                }
                for role, info in self.active_processes.items()
            },
            "standby_processes": {
                role.name: {
                    "pid": info.process.pid,
                    "state": info.state.value,
                    "creation_time": info.creation_time,
                }
                for role, info in self.standby_processes.items()
            },
        }