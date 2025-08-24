from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import TYPE_CHECKING, Callable, Optional, Tuple, TypeGuard

import torch

from sglang.srt.layers.moe.token_dispatcher import (
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonRunnerInput,
        TritonRunnerOutput,
    )


@dataclass
class MoeRunnerConfig:

    # MoE parameters
    num_experts: Optional[int] = None
    num_local_experts: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size_per_partition: Optional[int] = None
    layer_id: Optional[int] = None
    top_k: Optional[int] = None
    num_fused_shared_experts: Optional[int] = None
    params_dtype: Optional[torch.dtype] = None

    # Runner configuration
    activation: str = "silu"
    apply_router_weight_on_input: bool = False
    inplace: bool = True
    no_combine: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None


class RunnerInputFormat(Enum):
    TRITON = "triton"


class RunnerOutputFormat(Enum):
    TRITON = "triton"


@dataclass
class RunnerInput(ABC):

    @abstractmethod
    def get_format(self) -> RunnerInputFormat:
        pass

    def format_is_triton(self) -> TypeGuard[TritonRunnerInput]:
        return self.get_format() == RunnerInputFormat.TRITON


@dataclass
class MoeQuantInfo(ABC):
    """Moe quantication data."""

    pass


class RunnerOutput(ABC):

    @abstractmethod
    def get_format(self) -> RunnerOutputFormat:
        pass

    def format_is_triton(self) -> TypeGuard[TritonRunnerOutput]:
        return self.get_format() == RunnerOutputFormat.TRITON


class MoeRunnerCore(ABC):

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    @abstractmethod
    def run(
        self, runner_input: RunnerInput, quant_info: MoeQuantInfo, running_state: dict
    ) -> RunnerOutput:
        pass

    @property
    @abstractmethod
    def input_format(cls) -> RunnerInputFormat:
        pass

    @property
    @abstractmethod
    def output_format(cls) -> RunnerOutputFormat:
        pass


class FusedOpPool:

    _fused_ops: dict[str, Callable] = {}

    @classmethod
    def register_fused_func(
        cls, dispatch_name: str, runner_name: str, fused_func: Callable
    ):
        key = (dispatch_name, runner_name)
        if key in cls._fused_ops:
            raise ValueError(
                f"Fused function for {dispatch_name} to {runner_name} is already registered."
            )
        cls._fused_ops[key] = fused_func

    @classmethod
    def get_fused_func(cls, dispatch_name: str, runner_name: str) -> Optional[Callable]:
        key = (dispatch_name, runner_name)
        fused_func = cls._fused_ops.get(key)
        return fused_func


class PermuteMethodPool:

    _pre_permute_methods: dict[
        Tuple[DispatchOutputFormat, RunnerInputFormat], Callable
    ] = {}
    _post_permute_methods: dict[
        Tuple[RunnerOutputFormat, CombineInputFormat], Callable
    ] = {}

    @classmethod
    def register_pre_permute(
        cls,
        dispatch_output_name: str,
        runner_input_name: str,
        permute_func: Callable,
    ):
        """
        Register a customized pre-permute function for the given DispatchOutputFormat and RunnerInputFormat.

        :param dispatch_output_name: The DispatchOutputFormat name.
        :param runner_input_name: The RunnerInputFormat name.
        :param permute_func: The permute function to register.
        """
        key = (dispatch_output_name, runner_input_name)
        if key in cls._pre_permute_methods:
            raise ValueError(
                f"Pre-permute method for {dispatch_output_name} to {runner_input_name} is already registered."
            )
        cls._pre_permute_methods[key] = permute_func

    @classmethod
    def register_post_permute(
        cls,
        runner_output_name: str,
        combine_input_name: str,
        permute_func: Callable,
    ):
        """
        Register a customized post-permute function for the given RunnerOutputFormat and CombineInputFormat.

        :param runner_output_name: The RunnerOutputFormat name.
        :param combine_input_name: The CombineInputFormat name.
        :param permute_func: The permute function to register.
        """
        key = (runner_output_name, combine_input_name)
        if key in cls._post_permute_methods:
            raise ValueError(
                f"Post-permute method for {runner_output_name} to {combine_input_name} is already registered."
            )
        cls._post_permute_methods[key] = permute_func

    @classmethod
    def get_pre_permute(
        cls,
        dispatch_output_format: DispatchOutputFormat,
        runner_input_format: RunnerInputFormat,
    ) -> Callable:
        """
        Retrieve the pre-permute function for the given DispatchOutputFormat and RunnerInputFormat.

        :param dispatch_output_format: The DispatchOutputFormat type.
        :param runner_input_format: The RunnerInputFormat type.
        :return: The registered permute function or None if not found.
        """
        key = (dispatch_output_format, runner_input_format)
        pre_permute_func = cls._pre_permute_methods.get(key)
        assert (
            pre_permute_func is not None
        ), f"Pre-permute function for {dispatch_output_format} to {runner_input_format} is not registered"
        return pre_permute_func

    @classmethod
    def get_post_permute(
        cls,
        runner_output_format: RunnerOutputFormat,
        combine_input_format: CombineInputFormat,
    ) -> Callable:
        """
        Retrieve the post-permute function for the given RunnerOutputFormat and CombineInputFormat.

        :param runner_output_format: The RunnerOutputFormat type.
        :param combine_input_format: The CombineInputFormat type.
        :return: The registered permute function or None if not found.
        """
        key = (runner_output_format, combine_input_format)
        post_permute_func = cls._post_permute_methods.get(key)
        assert (
            post_permute_func is not None
        ), f"Post-permute function for {runner_output_format} to {combine_input_format} is not registered"
        return post_permute_func


def register_fused_func(
    dispatch_name: str,
    runner_name: str,
) -> Callable:
    """
    Decorator to register a fused function for the given DispatchOutputFormat and RunnerInputFormat.

    :param dispatch_name: The DispatchOutputFormat name.
    :param runner_name: The RunnerInputFormat name.
    :param fused_func: The fused function to register.
    :return: The decorator function.
    """

    def decorator(fused_func: Callable):
        FusedOpPool.register_fused_func(dispatch_name, runner_name, fused_func)
        return fused_func

    return decorator


def register_pre_permute(
    dispatch_output_name: str,
    runner_input_name: str,
) -> Callable:
    """
    Decorator to register a pre-permute function for the given DispatchOutputFormat and RunnerInputFormat.

    :param dispatch_output_name: The DispatchOutputFormat name.
    :param runner_input_name: The RunnerInputFormat name.
    :return: The decorator function.
    """

    def decorator(
        permute_func: Callable[
            [DispatchOutput, MoeQuantInfo, MoeRunnerConfig, dict], RunnerInput
        ]
    ) -> Callable:

        PermuteMethodPool.register_pre_permute(
            dispatch_output_name, runner_input_name, permute_func
        )
        return permute_func

    return decorator


def register_post_permute(
    runner_output_name: str,
    combine_input_name: str,
) -> Callable:
    """
    Decorator to register a post-permute function for the given RunnerOutputFormat and CombineInputFormat.

    :param runner_output_name: The RunnerOutputFormat name.
    :param combine_input_name: The CombineInputFormat name.
    :return: The decorator function.
    """

    def decorator(
        permute_func: Callable[
            [RunnerOutput, MoeQuantInfo, MoeRunnerConfig, dict], CombineInput
        ]
    ) -> Callable:
        PermuteMethodPool.register_post_permute(
            runner_output_name, combine_input_name, permute_func
        )
        return permute_func

    return decorator
