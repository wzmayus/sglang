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
from sglang.semi_pd.resident_process_manager import (
    ResidentProcessManager,
    DelayedSwitchingController,
    AsynchronousSwitchingController,
)
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs

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
        port_args: SemiPDPortArgs,
        initial_sm_allocation: SMAllocation,
        gpu_id: int = 0,
        tp_rank: int = 0,
        dp_rank: Optional[int] = None,
    ):
        self.server_args = server_args
        self.port_args = port_args
        self.current_sm_allocation = initial_sm_allocation
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        
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
        
        # 常驻进程管理器 - 论文4.3节核心组件
        self.resident_manager = ResidentProcessManager(
            server_args=server_args,
            port_args=port_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
        )
        
        # 延迟切换控制器 - 隐藏IPC和初始化延迟
        self.delayed_switching = DelayedSwitchingController(
            preparation_timeout=30.0
        )
        
        # 异步切换控制器 - 确保系统中始终有进程运行
        self.async_switching = AsynchronousSwitchingController()
        
        # 监控线程
        self.monitor_thread = None
        self.running = False
        
    def start(self):
        """启动进程轮换管理器"""
        logger.info("Starting process rotation manager with resident process support")
        self.running = True
        
        # 首先启动常驻进程管理器
        if not self.resident_manager.start():
            logger.error("Failed to start resident process manager")
            return False
            
        # 等待常驻进程就绪
        max_wait_time = 60.0  # 最大等待60秒
        wait_start = time.time()
        while not self.resident_manager.is_ready():
            if time.time() - wait_start > max_wait_time:
                logger.error("Timeout waiting for resident process to be ready")
                return False
            time.sleep(1.0)
            
        logger.info("Resident process is ready, starting initial worker processes")
        
        # 启动初始进程组
        self._start_initial_processes()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_processes)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Process rotation manager started successfully")
        return True
        
    def stop(self):
        """停止进程轮换管理器"""
        logger.info("Stopping process rotation manager")
        
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        # 终止所有工作进程
        self._terminate_all_processes()
        
        # 停止常驻进程管理器
        self.resident_manager.stop()
        
        self.context.term()
        
        logger.info("Process rotation manager stopped")
        
    def request_sm_reallocation(self, new_allocation: SMAllocation) -> bool:
        """
        请求SM重新分配 - 实现论文4.3节的延迟切换和异步切换机制
        
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
                f"Requesting SM reallocation with delayed switching: "
                f"P:{new_allocation.prefill_percentage}% "
                f"D:{new_allocation.decode_percentage}%"
            )
            
            self.target_sm_allocation = new_allocation
            self.switch_requested = True
            
            # 使用延迟切换机制 - 论文4.3节
            def preparation_callback():
                """准备新进程的回调函数"""
                try:
                    logger.info("Starting delayed switching preparation")
                    
                    # 启动新的休眠进程组，使用常驻进程的IPC信息
                    success = self._start_standby_processes_with_resident_ipc(new_allocation)
                    
                    if success:
                        logger.info("Standby processes prepared successfully")
                        return True
                    else:
                        logger.error("Failed to prepare standby processes")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error during preparation: {e}")
                    return False
            
            # 启动延迟切换
            success = self.delayed_switching.request_delayed_switch(preparation_callback)
            
            if success:
                # 通知活跃进程准备切换
                self._notify_active_processes_prepare_switch()
                logger.info("Delayed switching initiated successfully")
            else:
                self.switch_requested = False
                self.target_sm_allocation = None
                logger.error("Failed to initiate delayed switching")
                
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
                
    def _start_standby_processes_with_resident_ipc(self, sm_allocation: SMAllocation) -> bool:
        """启动休眠进程组，使用常驻进程的IPC信息"""
        logger.info("Starting standby process group with resident IPC")
        
        try:
            for role in [InstanceRole.PREFILL, InstanceRole.DECODE]:
                # 如果已有休眠进程，先终止
                if role in self.standby_processes:
                    self._terminate_process(self.standby_processes[role])
                    del self.standby_processes[role]
                    
                # 创建新的休眠进程，使用常驻进程的IPC信息
                process_info = self._create_process(role, sm_allocation, use_resident_ipc=True)
                if process_info:
                    process_info.state = ProcessState.STANDBY
                    self.standby_processes[role] = process_info
                    logger.info(f"Started standby {role.name} process with resident IPC")
                else:
                    logger.error(f"Failed to start standby {role.name} process")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error starting standby processes with resident IPC: {e}")
            return False
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
        sm_allocation: SMAllocation,
        use_resident_ipc: bool = True
    ) -> Optional[ProcessInfo]:
        """创建新进程 - 支持使用常驻进程的IPC信息"""
        try:
            # 设置SM分配环境变量
            if role == InstanceRole.PREFILL:
                sm_percentage = sm_allocation.prefill_percentage
            else:
                sm_percentage = sm_allocation.decode_percentage
                
            # 创建进程
            reader, writer = mp.Pipe(duplex=False)
            
            # 获取IPC信息队列
            ipc_info_queue = None
            if use_resident_ipc:
                if role == InstanceRole.PREFILL:
                    ipc_info = self.resident_manager.get_ipc_info_for_prefill()
                else:
                    ipc_info = self.resident_manager.get_ipc_info_for_decode()
                    
                if ipc_info:
                    ipc_info_queue = mp.Queue()
                    ipc_info_queue.put(ipc_info)
                else:
                    logger.warning(f"No IPC info available for {role.name}, creating without resident process")
            
            process = mp.Process(
                target=self._process_wrapper,
                args=(
                    run_scheduler_process,
                    self.server_args,
                    self.port_args,
                    self.gpu_id,
                    self.tp_rank,
                    0,  # moe_ep_rank
                    0,  # pp_rank
                    self.dp_rank,
                    writer,
                    ipc_info_queue,
                    use_resident_ipc,  # bypass_load_weight
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
        """执行进程切换 - 实现论文4.3节的异步切换机制"""
        logger.info("Performing asynchronous process switch")
        
        try:
            # 等待延迟切换准备完成
            if not self.delayed_switching.wait_for_switch_ready(timeout=30.0):
                logger.error("Delayed switching preparation timeout")
                self._rollback_switch()
                return
                
            logger.info("Delayed switching preparation completed, starting asynchronous switch")
            
            # 执行异步切换 - 论文4.3节核心机制
            old_active = self.active_processes.copy()
            
            # 为每个角色启动异步切换
            for role, standby_info in self.standby_processes.items():
                if role in old_active:
                    old_process = old_active[role].process
                    new_process = standby_info.process
                    
                    # 定义迭代完成回调
                    def iteration_complete_callback():
                        logger.info(f"Waiting for {role.name} iteration to complete")
                        # 这里应该等待当前迭代完成的信号
                        # 简化实现，等待一小段时间
                        time.sleep(0.1)
                    
                    # 启动异步切换
                    success = self.async_switching.start_asynchronous_switch(
                        role=role,
                        old_process=old_process,
                        new_process=new_process,
                        iteration_complete_callback=iteration_complete_callback
                    )
                    
                    if not success:
                        logger.error(f"Failed to start asynchronous switch for {role.name}")
                        self._rollback_switch()
                        return
            
            # 等待所有异步切换完成
            all_switches_completed = True
            for role in self.standby_processes.keys():
                if not self.async_switching.wait_for_switch_completion(role, timeout=30.0):
                    logger.error(f"Asynchronous switch timeout for {role.name}")
                    all_switches_completed = False
                    
            if not all_switches_completed:
                logger.error("Some asynchronous switches failed")
                self._rollback_switch()
                return
            
            # 将休眠进程提升为活跃进程
            for role, standby_info in self.standby_processes.items():
                standby_info.state = ProcessState.ACTIVE
                standby_info.last_switch_time = time.time()
                self.active_processes[role] = standby_info
                
            # 清空休眠进程
            self.standby_processes.clear()
            
            # 更新当前SM分配
            self.current_sm_allocation = self.target_sm_allocation
            
            # 重置切换状态
            self.switch_requested = False
            self.target_sm_allocation = None
            self.switch_event.set()
            
            logger.info("Asynchronous process switch completed successfully")
            
        except Exception as e:
            logger.error(f"Error during asynchronous process switch: {e}")
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
                    # 检查延迟切换是否准备完成
                    if self.delayed_switching.wait_for_switch_ready(timeout=0.1):
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
        status = {
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
            # 添加常驻进程管理器状态
            "resident_manager": self.resident_manager.get_status(),
            # 添加切换控制器状态
            "delayed_switching": {
                "switch_requested": self.delayed_switching.switch_requested,
                "preparation_complete": self.delayed_switching.preparation_complete.is_set(),
            },
            "async_switching": {
                "active_switches": list(self.async_switching.active_switches.keys()),
            },
        }
        
        return status