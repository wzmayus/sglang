# Copyright 2025 SGLang Team
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
"""TokenizerManagerFunctionMixin is a mixin for some non-core functions in TokenizerManager."""


import asyncio
import logging
import os
import pickle
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import fastapi

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    ExpertDistributionReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    ProfileReq,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import ReqState


logger = logging.getLogger(__name__)


class TokenizerManagerFunctionMixin:
    """
    A mixin for some non-core functions in TokenizerManager.

    We extract these non-core functions into a separate file to keep the original TokenizerManager shorter.
    The non-core functions include profiling, expert distribution, request dumping, LoRA, and similar stuff.
    """

    async def start_profile(
        self,
        output_dir: Optional[str] = None,
        start_step: Optional[int] = None,
        num_steps: Optional[int] = None,
        activities: Optional[List[str]] = None,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
        profile_by_stage: bool = False,
    ):
        self.auto_create_handle_loop()
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        with_stack = False if with_stack is False or env_with_stack is False else True
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_by_stage=profile_by_stage,
            profile_id=str(time.time()),
        )
        return await self._execute_profile(req)

    async def stop_profile(self):
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def start_expert_distribution_record(self):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.START_RECORD)

    async def stop_expert_distribution_record(self):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.STOP_RECORD)

    async def dump_expert_distribution_record(self):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.DUMP_RECORD)

    async def load_lora_adapter(
        self,
        obj: LoadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",
                obj.lora_name,
                obj.lora_path,
            )

            async with self.lora_update_lock:
                if (
                    self.server_args.max_loaded_loras is not None
                    and self.lora_registry.num_registered_loras
                    >= self.server_args.max_loaded_loras
                ):
                    raise ValueError(
                        f"Cannot load LoRA adapter {obj.lora_name} at path {obj.lora_path}. "
                        f"Maximum number of loaded LoRA adapters is {self.server_args.max_loaded_loras}. "
                        "Please unload some LoRA adapters before loading new ones."
                    )

                # Generate new uniquely identifiable LoRARef object.
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path=obj.lora_path,
                    pinned=obj.pinned,
                )

                # Trigger the actual loading operation at the backend processes.
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                # Register the LoRA adapter only after loading is successful.
                if result.success:
                    await self.lora_registry.register(new_adapter)

                return result
        except ValueError as e:
            return LoadLoRAAdapterReqOutput(
                success=False,
                error_message=str(e),
            )

    async def unload_lora_adapter(
        self,
        obj: UnloadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert (
                obj.lora_name is not None
            ), "lora_name must be provided to unload LoRA adapter"

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start unload Lora adapter. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                # Unregister the LoRA adapter from the registry to stop new requests for this adapter
                # from being started.
                lora_id = await self.lora_registry.unregister(obj.lora_name)
                obj.lora_id = lora_id

                # Initiate the actual unloading operation at the backend processes only after all
                # ongoing requests using this LoRA adapter are finished.
                await self.lora_registry.wait_for_unload(lora_id)
                result = (await self.update_lora_adapter_communicator(obj))[0]

                return result
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def get_weights_by_name(
        self, obj: GetWeightsByNameReqInput, request: Optional[fastapi.Request] = None
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if self.server_args.dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def slow_down(
        self,
        obj: SlowDownReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.slow_down_communicator(obj)

    async def open_session(
        self, obj: OpenSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        self.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(
        self, obj: CloseSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        await self.send_to_scheduler.send_pyobj(obj)

    async def get_internal_state(self) -> List[Dict[Any, Any]]:
        req = GetInternalStateReq()
        responses: List[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)
        )
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self, obj: SetInternalStateReq
    ) -> SetInternalStateReqOutput:
        responses: List[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)
        )
        return [res.internal_state for res in responses]

    async def get_load(self) -> dict:
        # TODO(lsyin): fake load report server
        if not self.current_load_lock.locked():
            async with self.current_load_lock:
                internal_state = await self.get_internal_state()
                self.current_load = internal_state[0]["load"]
        return {"load": self.current_load}

    def dump_requests(self, state: ReqState, out_dict: dict):
        self.dump_request_list.append(
            (state.obj, out_dict, state.created_time, time.time())
        )

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            self._dump_data_to_file(
                data_list=self.dump_request_list,
                filename=filename,
                log_message=f"Dump {len(self.dump_request_list)} requests to {filename}",
            )
            self.dump_request_list = []

    def record_request_for_crash_dump(self, state: ReqState, out_dict: dict):
        current_time = time.time()
        self.crash_dump_request_list.append(
            (state.obj, out_dict, state.created_time, current_time)
        )
        # Remove requests older than 5 minutes based on finish time
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300
        ):
            self.crash_dump_request_list.popleft()

    def _dump_data_to_file(
        self, data_list: List[Tuple], filename: str, log_message: str
    ):
        logger.info(log_message)
        to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_list.copy(),
        }

        def background_task():
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(to_dump_with_server_args, f)

        asyncio.create_task(asyncio.to_thread(background_task))

    def dump_requests_before_crash(self):
        if self.crash_dump_performed:
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return

        if not self.crash_dump_folder:
            return

        logger.error(f"Dumping requests before crash. {self.crash_dump_folder=}")
        self.crash_dump_performed = True

        # Check if NFS directory is available
        # expected_nfs_dir = "/" + self.crash_dump_folder.lstrip("/").split("/")[0]
        # use_nfs_dir = os.path.isdir(expected_nfs_dir) and os.access(
        #     expected_nfs_dir, os.W_OK
        # )
        use_nfs_dir = False
        if not use_nfs_dir:
            logger.error(
                f"Expected NFS directory is not available or writable. Uploading to GCS."
            )

        data_to_dump = []
        if self.crash_dump_request_list:
            data_to_dump.extend(self.crash_dump_request_list)

        # Add unfinished requests from rid_to_state
        unfinished_requests = []
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                unfinished_requests.append(
                    (
                        state.obj,
                        state.out_list[-1] if state.out_list else {},
                        state.created_time,
                        time.time(),
                    )
                )
        if unfinished_requests:
            data_to_dump.extend(unfinished_requests)

        if not data_to_dump:
            return

        object_name = f'crash_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'
        filename = os.path.join(
            self.crash_dump_folder,
            os.getenv("HOSTNAME", None),
            object_name,
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Include server_args in the dump
        data_to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_to_dump,
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_dump_with_server_args, f)
        logger.error(
            f"Dumped {len(self.crash_dump_request_list)} finished and {len(unfinished_requests)} unfinished requests before crash to {filename}"
        )

        def _upload_file_to_gcs(bucket_name, source_file_path, object_name):
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.upload_from_filename(source_file_path, if_generation_match=0)
            logger.error(
                f"Successfully uploaded {source_file_path} to gs://{bucket_name}/{object_name}"
            )

        if not use_nfs_dir:
            _upload_file_to_gcs(
                "sglang_crash_dump",
                filename,
                os.getenv("HOSTNAME", None) + "/" + object_name,
            )
