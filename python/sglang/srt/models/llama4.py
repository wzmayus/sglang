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

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/models/llama4.py
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Llama4TextConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.dp_attention import (
    dp_gather_partial,
    dp_scatter,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.models.llama import LlamaForCausalLM, LlamaMLP
from sglang.srt.utils import (
    add_prefix,
    fast_topk,
    get_compiler_backend,
    is_cuda,
    make_layers,
    DeepEPMode,
    is_non_idle_and_non_empty,
)

from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.managers.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.layers.moe.topk import _mask_topk_ids_padded_region

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


class Llama4MoE(nn.Module):

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_scores_aK, router_indices_aK = fast_topk(gating_output, topk, dim=-1)
        router_scores_aK = torch.sigmoid(router_scores_aK.float()).to(torch.float32)
        router_indices_aK = router_indices_aK.to(torch.int32)
        router_indices_aK = topk_ids_logical_to_physical(router_indices_aK, expert_location_dispatch_info)
        _mask_topk_ids_padded_region(router_indices_aK, num_token_non_padded)
        return (
            router_scores_aK.view(-1).reshape(router_scores_aK.shape),
            router_indices_aK,
        )

    def __init__(
        self,
        layer_id: int,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.device_module = torch.get_device_module()
        self.layer_id = layer_id

        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("router", prefix),
        )

        moe_config = dict()
        if global_server_args_dict["enable_deepep_moe"]:
            moe_config["deepep_mode"] = DeepEPMode[global_server_args_dict["deepep_mode"]]
        else:
            moe_config["apply_router_weight_on_input"] = True
        
        self.experts = get_moe_impl_class()(
            num_experts=config.num_local_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            custom_routing_function=Llama4MoE.custom_routing_function,
            renormalize=False,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            **moe_config,
        )

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            prefix=add_prefix("shared_expert", prefix),
            reduce_results=False,  # We need to do scatter before reduce
            **(
                dict(tp_rank=0, tp_size=1)
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = (
                config.num_local_experts + global_server_args_dict["ep_num_redundant_experts"]
            )
            self.top_k = config.num_experts_per_tok

            self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.num_local_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    def forward(self, hidden_states, forward_batch: ForwardBatch):
        if not global_server_args_dict["enable_deepep_moe"]:
            return self.forward_normal(hidden_states, forward_batch)
        else:
            return self.forward_deepep(hidden_states, forward_batch)
    
    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward_deepep(self, hidden_states, forward_batch: ForwardBatch):
        forward_mode = forward_batch.forward_mode
        shared_out = None
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.router(hidden_states)
            shared_out = self.shared_expert(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=False,
                renormalize=False,
                custom_routing_function=Llama4MoE.custom_routing_function,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
            # (yizhang2077) it is a hacky way for llama4, llama4 multiplies topk-weight after the 1st groupedgemm in MoE, 
            # while other MoE use to topk-weight after the 2nd groupedgemm in MoE. One of simple way is to change
            # hidden_states and topk_weights here, it can work since topk is 1 in llama4 models
            hidden_states = (hidden_states * topk_weights).to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights)
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )
        if self.ep_size > 1:
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                num_recv_tokens_per_expert,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states=hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_mode=forward_mode,
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            forward_mode=forward_mode,
        )
        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                hidden_states=final_hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_mode=forward_mode,
            )

        if shared_out is not None:
            final_hidden_states = final_hidden_states + shared_out

        return final_hidden_states

    def forward_normal(self, hidden_states, forward_batch: ForwardBatch):
        shared_out, routed_out = self._forward_core(
            hidden_states, forward_batch.forward_mode
        )

        out_aD = routed_out + shared_out

        if self.tp_size > 1:
            out_aD = tensor_model_parallel_all_reduce(out_aD)

        return out_aD

    def _forward_core(self, hidden_states, forward_mode: ForwardMode):
        if hidden_states.shape[0] < 4 and _is_cuda:
            return self._forward_core_shared_routed_overlap(hidden_states)
        else:
            return self._forward_core_normal(hidden_states)

    def _forward_core_normal(self, hidden_states):
        # router_scores: [num_tokens, num_experts]
        router_logits, _ = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        routed_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_out, routed_out

    def _forward_core_shared_routed_overlap(self, hidden_states):
        alt_stream = _get_or_create_alt_stream(self.device_module)

        alt_stream.wait_stream(self.device_module.current_stream())

        shared_out = self.shared_expert(hidden_states)

        with self.device_module.stream(alt_stream):
            # router_scores: [num_tokens, num_experts]
            router_logits, _ = self.router(hidden_states)
            routed_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        self.device_module.current_stream().wait_stream(alt_stream)

        return shared_out, routed_out

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits, _ = self.router(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)
        else:
            state.shared_output = None
    
    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            state.topk_weights_local, state.topk_idx_local = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=False,
                renormalize=False,
                custom_routing_function=Llama4MoE.custom_routing_function,
                num_token_non_padded=state.forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
            # (yizhang2077) it is a hacky way for llama4, llama4 multiplies topk-weight after the 1st groupedgemm in MoE, 
            # while other MoE use to topk-weight after the 2nd groupedgemm in MoE. One of simple way is to change
            # hidden_states and topk_weights here, it can work since topk is 1 in llama4 models
            state.hidden_states_mlp_input = (hidden_states * state.topk_weights_local).to(hidden_states.dtype)
            state.topk_weights_local = torch.ones_like(state.topk_weights_local)
        else:
            state.topk_idx_local = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            state.topk_weights_local = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            self.deepep_dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_idx=state.pop("topk_idx_local"),
                topk_weights=state.pop("topk_weights_local"),
                forward_mode=state.forward_batch.forward_mode,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                (
                    state.hidden_states_experts_input,
                    state.topk_idx_dispatched,
                    state.topk_weights_dispatched,
                    state.reorder_topk_ids,
                    state.num_recv_tokens_per_expert,
                    state.seg_indptr,
                    state.masked_m,
                    state.expected_m,
                ) = self.deepep_dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.hidden_states_experts_output = self.experts(
            hidden_states=state.pop("hidden_states_experts_input"),
            topk_idx=state.topk_idx_dispatched,
            topk_weights=state.topk_weights_dispatched,
            reorder_topk_ids=state.pop("reorder_topk_ids"),
            seg_indptr=state.pop("seg_indptr"),
            masked_m=state.pop("masked_m"),
            expected_m=state.pop("expected_m"),
            num_recv_tokens_per_expert=state.pop("num_recv_tokens_per_expert"),
            forward_mode=state.forward_batch.forward_mode,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.deepep_dispatcher.combine_a(
                hidden_states=state.pop("hidden_states_experts_output"),
                topk_idx=state.pop("topk_idx_dispatched"),
                topk_weights=state.pop("topk_weights_dispatched"),
                forward_mode=state.forward_batch.forward_mode,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.deepep_dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            x.add_(final_hidden_states)
            final_hidden_states = x

        state.hidden_states_mlp_output = final_hidden_states

_alt_stream = None


def _get_or_create_alt_stream(device_module):
    global _alt_stream
    if _alt_stream is None:
        _alt_stream = device_module.Stream()
    return _alt_stream


class Llama4Attention(nn.Module):

    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.use_rope = int((layer_id + 1) % 4 != 0)
        self.use_qk_norm = config.use_qk_norm and self.use_rope

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
            )
            if self.use_qk_norm
            else None
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
        )
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type in ["llama", "llama4"]:
            is_neox_style = False

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_theta),
                rope_scaling=rope_scaling if rope_scaling != "default" else None,
                is_neox_style=is_neox_style,
            )
            if self.use_rope
            else None
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
        )

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return attn_scale.unsqueeze(-1)

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _mul_attn_scale(self, positions, q):
        attn_scale = self._get_attn_scale(positions)
        return (q * attn_scale).to(q.dtype)

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None

        qkv, _ = self.qkv_proj(hidden_states)

        qk, v = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)

        if self.rotary_emb is not None:
            q_view, k_view = qk.split([self.q_size, self.kv_size], dim=-1)
            q_out_unused, k_out_unused = self.rotary_emb(positions, q_view, k_view)
            del q_view, k_view, q_out_unused, k_out_unused

        if self.qk_norm is not None:
            # TODO there are still 2 redundant direct_copy_kernel_cuda for this `reshape` and (in attn backend) q.contiguous(), maybe we can fuse them later
            qk = qk.reshape(-1, self.head_dim).contiguous().bfloat16()
            qk = self.qk_norm(qk).to(torch.bfloat16)
            qk = qk.reshape(-1, self.q_size + self.kv_size)

        q, k = qk.split([self.q_size, self.kv_size], dim=-1)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        if self.attn_temperature_tuning and not self.use_rope:
            q = self._mul_attn_scale(positions=positions, q=q)
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(*inner_state)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.local_dp_size = get_local_attention_dp_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.self_attn = Llama4Attention(
            config=config,
            layer_id=layer_id,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            prefix=add_prefix("self_attn", prefix),
        )
        self.config = config
        self.is_moe_layer = self._is_moe_layer(layer_id)
        if self.is_moe_layer:
            self.feed_forward = Llama4MoE(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        is_previous_moe_layer = self._is_moe_layer(layer_id - 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_moe_layer,
            is_previous_layer_sparse=is_previous_moe_layer,
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )
    
    def _is_moe_layer(self, layer_id: int) -> bool:
        return (layer_id + 1) % self.config.interleave_moe_layer_step == 0

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        
        # Fully Connected
        hidden_states = self.feed_forward(hidden_states, forward_batch)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        state.hidden_states_mlp_output = self.mlp(
            hidden_states, state.forward_batch.forward_mode
        )
    
    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output

class Llama4Model(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Llama4DecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        aux_hidden_states = []

        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(len(self.layers)):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(hidden_states + residual)
                layer = self.layers[i]
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                    )
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states, _ = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Llama4ForCausalLM(LlamaForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def _init_model(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return Llama4Model(config, quant_config=quant_config, prefix=prefix)


EntryClass = [Llama4ForCausalLM]
