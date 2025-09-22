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
Resident Process Manager for Semi-PD

Based on paper section 4.3: "To solve the first overhead, we introduce a resident process 
to consistently hold the weights and KV cache during serving, avoiding the repeated loading 
of weights and copying of KV cache."
"""

import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import zmq

from sglang.semi_pd.utils import IPCInfo, InstanceRole, get_ipc_handle
from sglang.srt.managers.semi_pd_scheduler import (
    SemiPDStandaloneScheduler,
    run_standalone_scheduler_process,
)
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class ResidentProcessInfo:
    """常驻进程信息"""
    process: mp.Process
    ipc_info: Optional[IPCInfo] = None
    creation_time: float = 0.0
    last_heartbeat: float = 0.0
    is_ready: bool = False


class ResidentProcessManager:
    """
    常驻进程管理器
    
    根据论文4.3节实现：
    1. 持续持有权重和KV cache，避免进程切换时的重复加载
    2. 通过IPC共享内存指针给prefill和decode进程
    3. 支持延迟切换(delayed switching)，隐藏IPC和初始化延迟
    4. 实现异步切换(asynchronous switching)，确保系统中始终有进程在运行
    """
    
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: SemiPDPortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int] = None,
    ):
        self.server_args = server_args
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        
        # 常驻进程管理
        self.resident_process: Optional[ResidentProcessInfo] = None
        self.process_lock = threading.Lock()
        
        # IPC队列用于与常驻进程通信
        self.p_ipc_info_queue = mp.Queue()
        self.d_ipc_info_queue = mp.Queue()
        
        # 监控线程
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        
        # 心跳机制
        self.heartbeat_interval = 5.0  # 5秒心跳间隔
        self.heartbeat_timeout = 15.0  # 15秒超时
        
        logger.info("Resident process manager initialized")
        
    def start(self) -> bool:
        """启动常驻进程管理器"""
        logger.info("Starting resident process manager")
        
        try:
            self.running = True
            
            # 启动常驻进程
            if not self._start_resident_process():
                logger.error("Failed to start resident process")
                return False
                
            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("Resident process manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start resident process manager: {e}")
            self.stop()
            return False
            
    def stop(self):
        """停止常驻进程管理器"""
        logger.info("Stopping resident process manager")
        
        self.running = False
        
        # 停止监控线程
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            
        # 终止常驻进程
        self._terminate_resident_process()
        
        logger.info("Resident process manager stopped")
        
    def get_ipc_info_for_prefill(self) -> Optional[IPCInfo]:
        """获取prefill进程的IPC信息"""
        try:
            if not self.p_ipc_info_queue.empty():
                return self.p_ipc_info_queue.get_nowait()
            elif self.resident_process and self.resident_process.ipc_info:
                return self.resident_process.ipc_info
            else:
                logger.warning("No IPC info available for prefill process")
                return None
        except Exception as e:
            logger.error(f"Error getting IPC info for prefill: {e}")
            return None
            
    def get_ipc_info_for_decode(self) -> Optional[IPCInfo]:
        """获取decode进程的IPC信息"""
        try:
            if not self.d_ipc_info_queue.empty():
                return self.d_ipc_info_queue.get_nowait()
            elif self.resident_process and self.resident_process.ipc_info:
                return self.resident_process.ipc_info
            else:
                logger.warning("No IPC info available for decode process")
                return None
        except Exception as e:
            logger.error(f"Error getting IPC info for decode: {e}")
            return None
            
    def is_ready(self) -> bool:
        """检查常驻进程是否就绪"""
        with self.process_lock:
            return (self.resident_process is not None and 
                    self.resident_process.is_ready and
                    self.resident_process.process.is_alive())
                    
    def _start_resident_process(self) -> bool:
        """启动常驻进程"""
        logger.info("Starting resident process")
        
        try:
            with self.process_lock:
                # 如果已有常驻进程，先终止
                if self.resident_process:
                    self._terminate_resident_process()
                    
                # 创建管道用于进程通信
                reader, writer = mp.Pipe(duplex=False)
                
                # 启动常驻进程
                process = mp.Process(
                    target=run_standalone_scheduler_process,
                    args=(
                        self.server_args,
                        self.port_args,
                        self.gpu_id,
                        self.tp_rank,
                        self.dp_rank,
                        writer,
                        False,  # bypass_load_weight
                        self.p_ipc_info_queue,
                        self.d_ipc_info_queue,
                    ),
                )
                
                process.start()
                
                # 等待进程就绪信号
                try:
                    ready_data = reader.recv()
                    if ready_data.get("status") == "ready":
                        # 获取IPC信息
                        ipc_info = None
                        if not self.p_ipc_info_queue.empty():
                            ipc_info = self.p_ipc_info_queue.get_nowait()
                            # 将IPC信息也放回decode队列
                            self.d_ipc_info_queue.put(ipc_info)
                            
                        self.resident_process = ResidentProcessInfo(
                            process=process,
                            ipc_info=ipc_info,
                            creation_time=time.time(),
                            last_heartbeat=time.time(),
                            is_ready=True,
                        )
                        
                        logger.info(f"Resident process started successfully (PID: {process.pid})")
                        return True
                    else:
                        logger.error(f"Resident process failed to start: {ready_data}")
                        process.terminate()
                        return False
                        
                except Exception as e:
                    logger.error(f"Error waiting for resident process ready signal: {e}")
                    process.terminate()
                    return False
                    
        except Exception as e:
            logger.error(f"Error starting resident process: {e}")
            return False
            
    def _terminate_resident_process(self):
        """终止常驻进程"""
        if self.resident_process:
            logger.info("Terminating resident process")
            
            try:
                if self.resident_process.process.is_alive():
                    self.resident_process.process.terminate()
                    self.resident_process.process.join(timeout=10.0)
                    
                    if self.resident_process.process.is_alive():
                        logger.warning("Resident process did not terminate gracefully, killing")
                        self.resident_process.process.kill()
                        self.resident_process.process.join(timeout=5.0)
                        
                logger.info("Resident process terminated")
                
            except Exception as e:
                logger.error(f"Error terminating resident process: {e}")
            finally:
                self.resident_process = None
                
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._check_resident_process_health()
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in resident process monitor loop: {e}")
                time.sleep(1.0)
                
    def _check_resident_process_health(self):
        """检查常驻进程健康状态"""
        with self.process_lock:
            if not self.resident_process:
                logger.warning("No resident process found, attempting to restart")
                self._start_resident_process()
                return
                
            # 检查进程是否存活
            if not self.resident_process.process.is_alive():
                logger.error("Resident process died, restarting")
                self._terminate_resident_process()
                self._start_resident_process()
                return
                
            # 更新心跳时间
            current_time = time.time()
            self.resident_process.last_heartbeat = current_time
            
            # 检查是否超时（这里简化实现，实际应该有进程间心跳机制）
            if (current_time - self.resident_process.creation_time > self.heartbeat_timeout and
                not self.resident_process.is_ready):
                logger.warning("Resident process startup timeout, restarting")
                self._terminate_resident_process()
                self._start_resident_process()
                
    def get_status(self) -> Dict:
        """获取常驻进程管理器状态"""
        with self.process_lock:
            if self.resident_process:
                return {
                    "running": self.running,
                    "resident_process": {
                        "pid": self.resident_process.process.pid,
                        "is_alive": self.resident_process.process.is_alive(),
                        "is_ready": self.resident_process.is_ready,
                        "creation_time": self.resident_process.creation_time,
                        "last_heartbeat": self.resident_process.last_heartbeat,
                        "uptime": time.time() - self.resident_process.creation_time,
                    },
                    "ipc_queues": {
                        "prefill_queue_size": self.p_ipc_info_queue.qsize(),
                        "decode_queue_size": self.d_ipc_info_queue.qsize(),
                    },
                }
            else:
                return {
                    "running": self.running,
                    "resident_process": None,
                    "ipc_queues": {
                        "prefill_queue_size": self.p_ipc_info_queue.qsize(),
                        "decode_queue_size": self.d_ipc_info_queue.qsize(),
                    },
                }


class DelayedSwitchingController:
    """
    延迟切换控制器
    
    根据论文4.3节实现延迟切换机制：
    "To hide the latency of IPC and initialization, semi-PD conducts the delayed switching, 
    running under the new (x,y) only when the preparation step finishes."
    """
    
    def __init__(self, preparation_timeout: float = 30.0):
        self.preparation_timeout = preparation_timeout
        self.switch_lock = threading.Lock()
        self.preparation_complete = threading.Event()
        self.switch_requested = False
        
    def request_delayed_switch(self, preparation_callback) -> bool:
        """
        请求延迟切换
        
        Args:
            preparation_callback: 准备工作的回调函数
            
        Returns:
            bool: 是否成功启动切换流程
        """
        with self.switch_lock:
            if self.switch_requested:
                logger.warning("Delayed switch already in progress")
                return False
                
            self.switch_requested = True
            self.preparation_complete.clear()
            
            # 在后台线程中执行准备工作
            preparation_thread = threading.Thread(
                target=self._execute_preparation,
                args=(preparation_callback,)
            )
            preparation_thread.daemon = True
            preparation_thread.start()
            
            return True
            
    def wait_for_switch_ready(self, timeout: Optional[float] = None) -> bool:
        """
        等待切换准备完成
        
        Args:
            timeout: 超时时间，None表示使用默认超时
            
        Returns:
            bool: 是否准备完成
        """
        if timeout is None:
            timeout = self.preparation_timeout
            
        return self.preparation_complete.wait(timeout)
        
    def _execute_preparation(self, preparation_callback):
        """执行准备工作"""
        try:
            logger.info("Starting delayed switch preparation")
            
            # 执行准备回调
            success = preparation_callback()
            
            if success:
                logger.info("Delayed switch preparation completed successfully")
                self.preparation_complete.set()
            else:
                logger.error("Delayed switch preparation failed")
                
        except Exception as e:
            logger.error(f"Error during delayed switch preparation: {e}")
        finally:
            with self.switch_lock:
                self.switch_requested = False


class AsynchronousSwitchingController:
    """
    异步切换控制器
    
    根据论文4.3节实现异步切换机制：
    "We can renew two MPS processes directly and only kill the worker who has finished its iteration. 
    Such an asynchronous behavior ensures there are always prefill and decode processes running in the system."
    """
    
    def __init__(self):
        self.switch_lock = threading.Lock()
        self.active_switches: Dict[InstanceRole, threading.Event] = {}
        
    def start_asynchronous_switch(
        self, 
        role: InstanceRole, 
        old_process: mp.Process,
        new_process: mp.Process,
        iteration_complete_callback
    ) -> bool:
        """
        启动异步切换
        
        Args:
            role: 进程角色
            old_process: 旧进程
            new_process: 新进程
            iteration_complete_callback: 迭代完成回调
            
        Returns:
            bool: 是否成功启动异步切换
        """
        with self.switch_lock:
            if role in self.active_switches:
                logger.warning(f"Asynchronous switch already active for {role.name}")
                return False
                
            # 创建切换事件
            switch_event = threading.Event()
            self.active_switches[role] = switch_event
            
            # 启动异步切换线程
            switch_thread = threading.Thread(
                target=self._execute_asynchronous_switch,
                args=(role, old_process, new_process, iteration_complete_callback, switch_event)
            )
            switch_thread.daemon = True
            switch_thread.start()
            
            logger.info(f"Started asynchronous switch for {role.name}")
            return True
            
    def _execute_asynchronous_switch(
        self, 
        role: InstanceRole,
        old_process: mp.Process,
        new_process: mp.Process,
        iteration_complete_callback,
        switch_event: threading.Event
    ):
        """执行异步切换"""
        try:
            logger.info(f"Executing asynchronous switch for {role.name}")
            
            # 等待当前迭代完成
            if iteration_complete_callback:
                iteration_complete_callback()
                
            # 终止旧进程
            if old_process.is_alive():
                old_process.terminate()
                old_process.join(timeout=5.0)
                
                if old_process.is_alive():
                    logger.warning(f"Old {role.name} process did not terminate gracefully, killing")
                    old_process.kill()
                    
            logger.info(f"Asynchronous switch completed for {role.name}")
            switch_event.set()
            
        except Exception as e:
            logger.error(f"Error during asynchronous switch for {role.name}: {e}")
        finally:
            with self.switch_lock:
                if role in self.active_switches:
                    del self.active_switches[role]
                    
    def wait_for_switch_completion(self, role: InstanceRole, timeout: float = 30.0) -> bool:
        """
        等待异步切换完成
        
        Args:
            role: 进程角色
            timeout: 超时时间
            
        Returns:
            bool: 是否切换完成
        """
        with self.switch_lock:
            if role not in self.active_switches:
                return True  # 没有活跃的切换
                
            switch_event = self.active_switches[role]
            
        return switch_event.wait(timeout)
        
    def is_switch_active(self, role: InstanceRole) -> bool:
        """检查是否有活跃的切换"""
        with self.switch_lock:
            return role in self.active_switches