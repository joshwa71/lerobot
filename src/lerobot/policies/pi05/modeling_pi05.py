#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
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

import builtins
import dataclasses
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

from ..modules.memory_lite import (
    MLPPlusMemory,
    TaskEmbeddingCache,
    aggregate_memory_losses,
    attach_memory_to_layer_list,
    checkpoint_recompute_context_fn,
    split_memory_params,
)
from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config


# Paligemma tokenizer id of "▁State" in pi05's fixed prompt template
# "Task: {instr}, State: {bins};\nAction: " (tokenizer name pinned in processor_pi05.py;
# a capitalized " State" cannot occur inside LIBERO instruction strings). The
# instruction/state boundary handed to the pooled-router modes (E45) is the index of the
# "," immediately before this marker; rows without the marker fall back to per-token
# routing inside the wrapper.
_VLM_STATE_MARKER_TOKEN_ID = 3040


def _vlm_instr_len_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
    is_m = tokens == _VLM_STATE_MARKER_TOKEN_ID
    has = is_m.any(dim=1)
    first = is_m.float().argmax(dim=1).long()
    return torch.where(has, first - 1, torch.zeros_like(first))


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
    task_emb=None,
    task_ids=None,
    router_x=None,
    vlm_router_x=None,
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Memory layers wrap the expert MLP only; pass language conditioning + task ids when present.
        if isinstance(layer.mlp, MLPPlusMemory):
            base_mlp = layer.mlp.mlp
            if base_mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(
                out_emb,
                lang_emb=task_emb,
                task_ids=task_ids,
                router_x=router_x if i == 1 else vlm_router_x,
            )
        else:
            # Convert to bfloat16 if the next layer (mlp) uses bfloat16
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


def compute_frozen_suffix_layer(
    layer_idx,
    prefix_hidden,
    frozen_hidden,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
    collect_mlp_input,
    run_mlp,
):
    """One expert-layer step of the FROZEN (memory-free) suffix stream.

    Used by memory_layer.use_frozen_base_input_features: downstream memory layers
    route (query + gate) on the backbone features as they would be WITHOUT any
    memory contribution, so the suffix stream is recomputed from the first memory
    layer onward with every memory module bypassed. The prefix stream never
    attends to the suffix (prefix-LM mask), so it is memory-independent and its
    per-layer hidden states are reused from the live pass for this layer's
    keys/values. Mirrors the expert-side computation of compute_layer_complete.

    Returns (mlp_input if collect_mlp_input else None, layer_output if run_mlp else None).
    """
    expert_layer = gemma_expert.model.layers[layer_idx]
    pg_attn = paligemma.model.language_model.layers[layer_idx].self_attn
    prefix_len = prefix_hidden.shape[1]

    # Prefix keys/values for this layer (queries not needed: prefix outputs come from the live pass).
    pre_normed, _ = layernorm_forward(
        paligemma.model.language_model.layers[layer_idx].input_layernorm, prefix_hidden, adarms_cond[0]
    )
    pre_shape = (*pre_normed.shape[:-1], -1, pg_attn.head_dim)
    k_pre = pg_attn.k_proj(pre_normed).view(pre_shape).transpose(1, 2)
    v_pre = pg_attn.v_proj(pre_normed).view(pre_shape).transpose(1, 2)

    # Frozen suffix q/k/v.
    fro_normed, attn_gate = layernorm_forward(
        expert_layer.input_layernorm, frozen_hidden, adarms_cond[1]
    )
    fro_shape = (*fro_normed.shape[:-1], -1, expert_layer.self_attn.head_dim)
    q_fro = expert_layer.self_attn.q_proj(fro_normed).view(fro_shape).transpose(1, 2)
    k_fro = expert_layer.self_attn.k_proj(fro_normed).view(fro_shape).transpose(1, 2)
    v_fro = expert_layer.self_attn.v_proj(fro_normed).view(fro_shape).transpose(1, 2)

    # Rotary embeddings: same cos/sin as the joint pass, applied per position slice.
    dummy_tensor = torch.zeros(
        q_fro.shape[0], position_ids.shape[1], q_fro.shape[-1], device=q_fro.device, dtype=q_fro.dtype
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    _, k_pre = modeling_gemma.apply_rotary_pos_emb(
        k_pre, k_pre, cos[:, :prefix_len], sin[:, :prefix_len], unsqueeze_dim=1
    )
    q_fro, k_fro = modeling_gemma.apply_rotary_pos_emb(
        q_fro, k_fro, cos[:, prefix_len:], sin[:, prefix_len:], unsqueeze_dim=1
    )

    key_states = torch.cat([k_pre, k_fro], dim=2)
    value_states = torch.cat([v_pre, v_fro], dim=2)

    # Suffix rows of the joint mask: same column semantics ([prefix, suffix]).
    att_output, _ = modeling_gemma.eager_attention_forward(
        pg_attn,
        q_fro,
        key_states,
        value_states,
        attention_mask[..., prefix_len:, :],
        pg_attn.scaling,
    )
    att_output = att_output.reshape(frozen_hidden.shape[0], -1, 1 * 8 * pg_attn.head_dim)
    if att_output.dtype != expert_layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(expert_layer.self_attn.o_proj.weight.dtype)
    out_emb = expert_layer.self_attn.o_proj(att_output)
    out_emb = _gated_residual(frozen_hidden, out_emb, attn_gate)
    after_first_residual = out_emb.clone()
    out_emb, mlp_gate = layernorm_forward(
        expert_layer.post_attention_layernorm, out_emb, adarms_cond[1]
    )

    base_mlp = expert_layer.mlp.mlp if isinstance(expert_layer.mlp, MLPPlusMemory) else expert_layer.mlp
    if base_mlp.up_proj.weight.dtype == torch.bfloat16:
        out_emb = out_emb.to(dtype=torch.bfloat16)
    mlp_input = out_emb if collect_mlp_input else None
    if not run_mlp:
        return mlp_input, None
    out = base_mlp(out_emb)
    out = _gated_residual(after_first_residual, out, mlp_gate)
    return mlp_input, out


def compute_frozen_prefix_layer(
    layer_idx,
    frozen_hidden,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    collect_mlp_input,
    run_mlp,
):
    """One paligemma-LM-layer step of the FROZEN (memory-free) PREFIX stream (E45).

    The VLM-side twin of compute_frozen_suffix_layer: when memory sits on more than
    one prefix layer, every VLM memory layer above the first must route (query + gate
    + pooled anchor keys) on the prefix features as they would be WITHOUT any memory
    contribution — otherwise value training at the lower layer re-points the upper
    layer's frozen router (the E38 routing-drift channel, one-layer edition). The
    prefix attends ONLY prefix positions (prefix-LM mask), so the frozen stream is
    fully self-contained: q/k/v all come from the frozen hidden and the joint mask's
    prefix rows x prefix columns slice loses nothing.

    Returns (mlp_input if collect_mlp_input else None, layer_output if run_mlp else None).
    """
    layer = paligemma.model.language_model.layers[layer_idx]
    prefix_len = frozen_hidden.shape[1]

    normed, attn_gate = layernorm_forward(layer.input_layernorm, frozen_hidden, adarms_cond[0])
    shape = (*normed.shape[:-1], -1, layer.self_attn.head_dim)
    q = layer.self_attn.q_proj(normed).view(shape).transpose(1, 2)
    k = layer.self_attn.k_proj(normed).view(shape).transpose(1, 2)
    v = layer.self_attn.v_proj(normed).view(shape).transpose(1, 2)

    dummy_tensor = torch.zeros(
        q.shape[0], position_ids.shape[1], q.shape[-1], device=q.device, dtype=q.dtype
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    q, k = modeling_gemma.apply_rotary_pos_emb(
        q, k, cos[:, :prefix_len], sin[:, :prefix_len], unsqueeze_dim=1
    )

    att_output, _ = modeling_gemma.eager_attention_forward(
        layer.self_attn,
        q,
        k,
        v,
        attention_mask[..., :prefix_len, :prefix_len],
        layer.self_attn.scaling,
    )
    att_output = att_output.reshape(frozen_hidden.shape[0], -1, 1 * 8 * layer.self_attn.head_dim)
    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
    out_emb = layer.self_attn.o_proj(att_output)
    out_emb = _gated_residual(frozen_hidden, out_emb, attn_gate)
    after_first_residual = out_emb.clone()
    out_emb, mlp_gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[0])

    base_mlp = layer.mlp.mlp if isinstance(layer.mlp, MLPPlusMemory) else layer.mlp
    if base_mlp.up_proj.weight.dtype == torch.bfloat16:
        out_emb = out_emb.to(dtype=torch.bfloat16)
    mlp_input = out_emb if collect_mlp_input else None
    if not run_mlp:
        return mlp_input, None
    out = base_mlp(out_emb)
    out = _gated_residual(after_first_residual, out, mlp_gate)
    return mlp_input, out


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def attach_memory_to_expert(self, cfg):
        """Attach product-key memory layers to the action expert MLP blocks.

        PI05 only attaches memory to the action expert (gemma_expert), not the
        VLM (paligemma) backbone.
        """
        target_layers = attach_memory_to_layer_list(
            self.gemma_expert.model.layers,
            dim=self.gemma_expert.config.hidden_size,
            cfg=cfg,
            label="EXPERT",
        )
        try:
            self.mem_target_layers = target_layers
        except Exception:
            pass
        self._mem_cfg = cfg
        self._mem_layer_indices = sorted(
            i for i, layer in enumerate(self.gemma_expert.model.layers) if isinstance(layer.mlp, MLPPlusMemory)
        )
        if getattr(cfg, "use_frozen_base_input_features", False):
            if getattr(cfg, "memory_only", False):
                raise ValueError("use_frozen_base_input_features is incompatible with memory_layer.memory_only")
            logging.info(
                f"Frozen-base routing ENABLED: memory layers {self._mem_layer_indices} route on the "
                "memory-free backbone features (dual-path)."
            )
        if getattr(cfg, "frozen_prepass", False):
            if not getattr(cfg, "use_frozen_base_input_features", False):
                raise ValueError(
                    "memory_layer.frozen_prepass requires use_frozen_base_input_features=true — it is "
                    "an implementation of the same routing-stationarity property (routing reads the "
                    "memory-free features), obtained via a full pre-pass instead of placement."
                )
            logging.info(
                "Frozen PRE-PASS ENABLED (E59): all routing inputs (expert router_x, VLM router_x, "
                "expert anchors, inference prefix KV for the expert's pass A) come from one "
                "memory-bypassed forward per batch; the VLM placement guard is lifted."
            )
        # ---- VLM-side text-span memory (E44): additional modules on the paligemma LM ----
        # Placed ABOVE the highest expert memory layer so the prefix KV the expert's frozen
        # routing branch consumes (layers <= max(expert layers)) is untouched; the lowest VLM
        # module's own router input is memory-free by construction. The joint training path
        # (compute_layer_complete) dispatches wrapped MLPs generically for BOTH towers, so
        # task_ids/lang_emb reach these modules with no extra threading; the prefix-only
        # inference path calls mlp(x) plain (losses off, memory active) — also correct.
        vlm_layers = list(getattr(cfg, "vlm_layers", []) or [])
        if vlm_layers:
            exp_max = max(self._mem_layer_indices) if self._mem_layer_indices else -1
            if min(vlm_layers) <= exp_max:
                if not getattr(cfg, "frozen_prepass", False):
                    raise ValueError(
                        f"vlm_layers {vlm_layers} must all sit ABOVE the highest expert memory layer "
                        f"({exp_max}) to preserve expert routing stationarity (prefix KV <= {exp_max}). "
                        "Set memory_layer.frozen_prepass=true to lift this constraint (routing inputs "
                        "then come from a full memory-free pre-pass; ~+1 forward/step)."
                    )
                logging.info(
                    f"INTERLEAVED memory placement (frozen_prepass): expert layers "
                    f"{self._mem_layer_indices} / VLM layers {vlm_layers} — expert routing, anchors, "
                    "and the inference pass-A prefix KV are served by the memory-free pre-pass."
                )
            vlm_cfg = dataclasses.replace(
                cfg,
                layers=vlm_layers,
                mem_n_keys=int(getattr(cfg, "vlm_mem_n_keys", 256)),
                lora_rank=int(getattr(cfg, "vlm_lora_rank", 2)),
                mem_knn=int(getattr(cfg, "vlm_mem_knn", 16)),
                # E45: the routing-loss candidate pool must match the tower's actual
                # retrieval set (the E14-16 alignment rule) — derive it per tower.
                routing_loss_topk=int(getattr(cfg, "vlm_mem_knn", 16)),
                layer_ranks=[],
                lang_to_query=False,
                use_frozen_base_input_features=False,
                # E59: the pre-pass is orchestrated at the policy level; the derived
                # per-tower cfg must not re-trigger the config-level validation
                # (frozen_prepass requires use_frozen_base_input_features, cleared above).
                frozen_prepass=False,
                text_span=int(getattr(cfg, "vlm_text_span", 200)),
                image_regions=int(getattr(cfg, "vlm_image_regions", 0) or 0),
                image_pool_weights=list(getattr(cfg, "vlm_image_pool_weights", [1.0, 0.5]) or [1.0, 0.5]),
                # E57: the VLM tower's per-layer value-input-noise sigmas ride the derived
                # cfg's expert-position field (matched to `layers`=vlm_layers by order).
                value_input_noise_sigma=list(getattr(cfg, "vlm_value_input_noise_sigma", []) or []),
                vlm_value_input_noise_sigma=[],
                vlm_layers=[],
                # E52: the expert-anchor mix is an expert-tower mechanism; the VLM
                # modules keep their own pooled-key machinery.
                expert_anchor_pool="",
                # E61: the VLM tower's share groups ride the derived cfg's expert-position
                # field (attach_memory_to_layer_list reads cfg.share_groups against the
                # tower it is wrapping); clear the vlm field to keep validation happy.
                share_groups=list(getattr(cfg, "vlm_share_groups", []) or []),
                vlm_share_groups=[],
            )
            vlm_targets = attach_memory_to_layer_list(
                self.paligemma.model.language_model.layers,
                dim=self.paligemma.config.text_config.hidden_size,
                cfg=vlm_cfg,
                label="VLM",
            )
            self._vlm_mem_layer_indices = sorted(
                i for i, layer in enumerate(self.paligemma.model.language_model.layers)
                if isinstance(layer.mlp, MLPPlusMemory)
            )
            logging.info(
                f"VLM text-span memory attached at LM layers {vlm_targets}: n_keys={vlm_cfg.mem_n_keys} "
                f"(bank {vlm_cfg.mem_n_keys ** 2}), r={vlm_cfg.lora_rank}, knn={vlm_cfg.mem_knn}, "
                f"span=last {vlm_cfg.text_span} positions (the tokenized language field)."
            )
        # ---- E52 expert-anchor pooled routing: per-layer pairing (expert layer j routes
        # on B*nrm(W_a @ pooled instr hidden at LM layer j) + (1-B)*nrm(token)). Capture
        # is a forward_pre_hook on the paired LM layer's mlp module — it fires with the
        # post-attn-LN mlp input (the router-input quantity) on BOTH the joint training
        # path (compute_layer_complete calls layer.mlp(out_emb); i=0 runs before i=1 in
        # each layer iteration, so the anchor is fresh when the expert wrapper consumes
        # it) and the inference prefix pass (HF forward). Anchor layers sit below
        # min(vlm_layers) (placement guard), so duplicate prefix passes (VLM frozen
        # pass A, grad-ckpt recompute) recompute identical values — overwrite-safe —
        # and the anchor is memory-free/stationary by construction. Values detached:
        # routing is a frozen function of the backbone; W_a carries the trainable
        # directions.
        if str(getattr(cfg, "expert_anchor_pool", "") or ""):
            lm_layers = self.paligemma.model.language_model.layers
            bad = [i for i in self._mem_layer_indices if i >= len(lm_layers)]
            if bad:
                raise ValueError(
                    f"expert_anchor_pool: expert memory layers {bad} have no paired LM layer"
                )
            src = int(getattr(cfg, "expert_anchor_src_dim", 2048))
            lm_dim = int(self.paligemma.config.text_config.hidden_size)
            if src != lm_dim:
                raise ValueError(
                    f"expert_anchor_src_dim={src} != paligemma LM hidden size {lm_dim}"
                )
            self._anchor_masks = None
            self._anchor_instr_len = None
            for j in self._mem_layer_indices:
                wrapper = self.gemma_expert.model.layers[j].mlp
                lm_layers[j].mlp.register_forward_pre_hook(self._make_anchor_hook(wrapper))
            logging.info(
                f"Expert-anchor routing ENABLED: expert layers {self._mem_layer_indices} route on "
                f"pooled LM instruction hiddens from the SAME LM layer indices "
                f"(B={float(getattr(cfg, 'expert_anchor_weight', 0.5))})."
            )

    def _make_anchor_hook(self, wrapper):
        """E52: capture hook for one (LM layer j -> expert layer j) anchor pairing."""

        def hook(mod, args):
            # E59 frozen_prepass: anchors are captured during the memory-free pre-pass and
            # LOCKED for the live pass — under interleaved placement the live prefix at the
            # anchor layer carries memory content, so a live-pass overwrite would break the
            # anchor's stationarity. Without the pre-pass the lock is never set and the
            # historical overwrite-safe behavior is unchanged.
            if getattr(self, "_anchor_lock", False):
                return
            x = args[0]
            masks = getattr(self, "_anchor_masks", None)
            il = getattr(self, "_anchor_instr_len", None)
            if masks is None or il is None or not torch.is_tensor(x) or x.dim() != 3:
                return
            n_lang = masks.shape[1]
            # Full-prefix calls only: (B, images+language, D). The suffix stream never
            # passes through LM mlps; other shapes (defensive) leave the anchor as-is.
            if x.shape[0] != masks.shape[0] or x.shape[1] < n_lang:
                return
            with torch.no_grad():
                lang = x[:, -n_lang:].float()
                ilc = il.to(x.device)
                vm = masks.to(device=x.device).bool()
                pos = torch.arange(n_lang, device=x.device).unsqueeze(0)
                # Instruction pool = positions [3, b): skips the constant "<bos> Task :"
                # prefix, ends at the "," before the State marker (E45 convention).
                imask = (pos >= 3) & (pos < ilc.unsqueeze(1)) & vm
                m = imask.unsqueeze(-1).float()
                pooled = (lang * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
                valid = (ilc > 4) & (imask.sum(dim=1) > 0)
            wrapper.set_expert_anchor(pooled.detach(), valid)

        return hook

    def set_vlm_token_mask(self, masks: torch.Tensor | None, instr_len: torch.Tensor | None = None,
                           img_active: torch.Tensor | None = None):
        """E44 pad fix: hand the language-field attention mask (B, tokenizer_max_length) to
        the VLM text-span memory wrappers so pad positions are excluded from memory output,
        usage statistics, TF counts, and the routing/contrastive losses.

        instr_len (E45, pooled state routing): per-sample field index of the "," preceding
        the "State" marker in pi05's "Task: {instr}, State: {bins};" prompt — the
        instruction/state boundary the pooled-router modes key on. None = per-token routing.

        img_active (E49, image-span pooled routing): (B, n_cam) bool — per camera-slot
        validity from prepare_images (empty_cameras slots are False). Real cameras are
        appended FIRST, so active image positions form a contiguous prefix of the image
        block. None disables the image span (modules fall back to text-span-only).
        """
        for i in getattr(self, "_vlm_mem_layer_indices", []) or []:
            mlp = self.paligemma.model.language_model.layers[i].mlp
            mlp._ctx_valid_mask = masks
            mlp._ctx_instr_len = instr_len
            mlp._ctx_img_active = img_active
        # E52 expert-anchor: the capture hooks (registered in attach_memory_to_expert)
        # read the language-field mask + instruction boundary from here. Both forward
        # entry points call this method before the prefix runs, so the ctx is fresh
        # per batch on every path.
        self._anchor_masks = masks
        self._anchor_instr_len = instr_len

    def _frozen_routing_enabled(self) -> bool:
        cfg = getattr(self, "_mem_cfg", None)
        return bool(
            cfg is not None
            and getattr(cfg, "use_frozen_base_input_features", False)
            and getattr(self, "_mem_layer_indices", None)
        )

    def _vlm_frozen_routing_enabled(self) -> bool:
        # Same flag as the expert side: "routing reads memory-free features" now covers
        # both towers. With a single VLM memory layer the fork is unnecessary (its router
        # input is memory-free by placement), so it activates only at >= 2 VLM layers.
        cfg = getattr(self, "_mem_cfg", None)
        vlm_idx = getattr(self, "_vlm_mem_layer_indices", None) or []
        return bool(
            cfg is not None
            and getattr(cfg, "use_frozen_base_input_features", False)
            and len(vlm_idx) >= 2
        )

    def _frozen_prepass_enabled(self) -> bool:
        # E59: the full memory-free pre-pass replaces BOTH lazy forks and serves every
        # routing input (expert router_x, VLM router_x, expert anchors, and — at
        # inference — the memory-free prefix KV the expert's pass A attends).
        cfg = getattr(self, "_mem_cfg", None)
        has_mem = bool(getattr(self, "_mem_layer_indices", None)) or bool(
            getattr(self, "_vlm_mem_layer_indices", None)
        )
        return bool(
            cfg is not None
            and getattr(cfg, "frozen_prepass", False)
            and getattr(cfg, "use_frozen_base_input_features", False)
            and has_mem
        )

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        task_emb: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            # Prefix-only pass (inference: builds the KV caches the denoise loop consumes).
            # VLM frozen routing (E45), pass A: run the prefix once with every VLM memory
            # bypassed so each VLM wrapper stashes its memory-free mlp input; the live pass
            # (B) pops the stash as router_x. Mirrors the suffix-side dual pass below. No
            # KV hygiene needed here: pass A runs cache-less and its output is discarded.
            #
            # E59 frozen_prepass extends pass A: it runs for ALL VLM wrappers (not just
            # >=2), keeps its KV cache (`_frozen_prefix_kv` — the memory-free prefix KV
            # the expert's suffix pass A must attend under interleaved placement), and
            # captures the expert anchors from the memory-free stream, locking them
            # against overwrite by the live prefix pass.
            prepass = self._frozen_prepass_enabled()
            if not prepass and getattr(self, "_anchor_lock", False):
                self._anchor_lock = False  # safety: never leave anchors locked without a pre-pass
            vlm_frozen = self._vlm_frozen_routing_enabled() or (
                prepass and bool(getattr(self, "_vlm_mem_layer_indices", None))
            )
            if vlm_frozen:
                if prepass:
                    self._anchor_lock = False
                vlm_wrappers = [
                    self.paligemma.model.language_model.layers[i].mlp
                    for i in self._vlm_mem_layer_indices
                ]
                for m in vlm_wrappers:
                    m.begin_frozen_capture()
                try:
                    with torch.no_grad():
                        pass_a_out = self.paligemma.model.language_model.forward(
                            inputs_embeds=inputs_embeds[0],
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            use_cache=bool(prepass),
                            adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                        )
                    if prepass:
                        self._frozen_prefix_kv = pass_a_out.past_key_values
                        self._frozen_prefix_kv_bs = int(inputs_embeds[0].shape[0])
                        self._anchor_lock = True
                except Exception:
                    for m in vlm_wrappers:
                        m._frozen_capture = False
                        m._frozen_stash = []
                    self._anchor_lock = False
                    raise
                for m in vlm_wrappers:
                    m.end_frozen_capture()
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            if vlm_frozen:
                for m in vlm_wrappers:
                    m.assert_frozen_stash_consumed()
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            # Suffix-only path used by denoise_step. PiGemma's decoder layer (pi_gemma.py)
            # pops lang_emb/task_ids out of **kwargs and dispatches to MLPPlusMemory when
            # the wrapped expert MLP is memory-augmented, so passing them here is what
            # restores the FiLM language conditioning at inference for pi05+memory.
            frozen_routing = self._frozen_routing_enabled()
            if frozen_routing:
                # Frozen-base routing, pass A: run the expert once with memory
                # bypassed so each memory layer stashes its memory-free mlp input
                # (the routing features). The prefix KV cache is deep-copied because
                # the attention appends suffix keys to it (denoise_step deep-copies
                # for the same reason).
                mem_wrappers = [
                    self.gemma_expert.model.layers[i].mlp for i in self._mem_layer_indices
                ]
                for m in mem_wrappers:
                    m.begin_frozen_capture()
                try:
                    with torch.no_grad():
                        # E59 frozen_prepass: under interleaved placement the LIVE prefix KV
                        # carries VLM memory content, so pass A must attend the memory-free
                        # prefix KV captured by the prefix pass A. Without the pre-pass (or
                        # with no VLM banks) the live KV is memory-free at expert layers by
                        # placement and remains the correct source.
                        pkv_src = past_key_values
                        if self._frozen_prepass_enabled():
                            fkv = getattr(self, "_frozen_prefix_kv", None)
                            if fkv is not None:
                                exp_bs = int(inputs_embeds[1].shape[0])
                                got_bs = int(getattr(self, "_frozen_prefix_kv_bs", -1))
                                if got_bs != exp_bs:
                                    raise RuntimeError(
                                        f"frozen_prepass: stale memory-free prefix KV (batch {got_bs} "
                                        f"vs suffix batch {exp_bs}) — prefix pass must precede denoise."
                                    )
                                pkv_src = fkv
                        pkv_a = copy.deepcopy(pkv_src) if pkv_src is not None else None
                        self.gemma_expert.model.forward(
                            inputs_embeds=inputs_embeds[1],
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=pkv_a,
                            use_cache=use_cache,
                            adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                            lang_emb=task_emb,
                            task_ids=task_ids,
                        )
                except Exception:
                    for m in mem_wrappers:
                        m._frozen_capture = False
                        m._frozen_stash = []
                    raise
                for m in mem_wrappers:
                    m.end_frozen_capture()
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                lang_emb=task_emb,
                task_ids=task_ids,
            )
            if frozen_routing:
                for m in mem_wrappers:
                    m.assert_frozen_stash_consumed()
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # E59 frozen_prepass (training/joint path): ONE memory-bypassed forward of the
            # whole network computes every routing input for this batch — each wrapper
            # stashes its memory-free mlp input (drained below into explicit router_x
            # args, which re-thread safely through gradient-checkpoint recompute), and
            # the E52 anchor hooks fire on the memory-free prefix stream and are then
            # LOCKED so the live pass cannot overwrite them. Replaces both lazy forks;
            # required for interleaved expert/VLM placement, where the live prefix at
            # expert layers carries VLM memory content.
            prepass = self._frozen_prepass_enabled()
            if not prepass and getattr(self, "_anchor_lock", False):
                self._anchor_lock = False  # safety: never leave anchors locked without a pre-pass
            prepass_rx: dict[int, torch.Tensor] = {}
            prepass_vrx: dict[int, torch.Tensor] = {}
            if prepass:
                self._anchor_lock = False
                exp_wr = [
                    (i, self.gemma_expert.model.layers[i].mlp)
                    for i in (getattr(self, "_mem_layer_indices", None) or [])
                ]
                vlm_wr = [
                    (i, self.paligemma.model.language_model.layers[i].mlp)
                    for i in (getattr(self, "_vlm_mem_layer_indices", None) or [])
                ]
                for _, m in exp_wr + vlm_wr:
                    m.begin_frozen_capture()
                try:
                    with torch.no_grad():
                        pe = inputs_embeds
                        for li in range(num_layers):
                            pe = compute_layer_complete(
                                li,
                                pe,
                                attention_mask,
                                position_ids,
                                adarms_cond,
                                paligemma=self.paligemma,
                                gemma_expert=self.gemma_expert,
                                task_emb=task_emb,
                                task_ids=task_ids,
                                router_x=None,
                                vlm_router_x=None,
                            )
                except Exception:
                    for _, m in exp_wr + vlm_wr:
                        m._frozen_capture = False
                        m._frozen_stash = []
                    self._anchor_lock = False
                    raise
                for i, m in exp_wr:
                    m.end_frozen_capture()
                    prepass_rx[i] = m.drain_frozen_stash()
                for i, m in vlm_wr:
                    m.end_frozen_capture()
                    prepass_vrx[i] = m.drain_frozen_stash()
                self._anchor_lock = True

            # Frozen-base routing (training/joint path): maintain a memory-free
            # suffix stream from the first memory layer to the last, whose per-layer
            # mlp inputs are the routing features for the live pass. Layers below the
            # first memory layer are identical in both streams (no memory to diverge
            # them), so the fork starts lazily at the first memory layer, and the
            # frozen stream is dropped once the last memory layer's routing features
            # are collected. Runs under no_grad: routing features are a frozen
            # function of the backbone by design. (Superseded by the pre-pass when
            # frozen_prepass is on.)
            frozen_routing = self._frozen_routing_enabled() and not prepass
            mem_idx = self._mem_layer_indices if frozen_routing else []
            fork_lo = mem_idx[0] if frozen_routing else -1
            fork_hi = mem_idx[-1] if frozen_routing else -1
            frozen_hidden = None
            # VLM-side frozen prefix routing (E45): same lazy-fork pattern on the prefix
            # tower. The first VLM memory layer's live mlp-input is memory-free by
            # placement (router_x stays None there); every later VLM memory layer routes
            # on the frozen prefix stream advanced with memory bypassed.
            vlm_frozen = self._vlm_frozen_routing_enabled() and not prepass
            vlm_idx = self._vlm_mem_layer_indices if vlm_frozen else []
            vlm_fork_lo = vlm_idx[0] if vlm_frozen else -1
            vlm_fork_hi = vlm_idx[-1] if vlm_frozen else -1
            vlm_frozen_hidden = None

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                router_x = prepass_rx.get(layer_idx) if prepass else None
                vlm_router_x = prepass_vrx.get(layer_idx) if prepass else None
                if vlm_frozen and vlm_fork_lo <= layer_idx <= vlm_fork_hi:
                    with torch.no_grad():
                        if layer_idx == vlm_fork_lo:
                            _, vlm_frozen_hidden = compute_frozen_prefix_layer(
                                layer_idx,
                                inputs_embeds[0],
                                attention_mask,
                                position_ids,
                                adarms_cond,
                                self.paligemma,
                                collect_mlp_input=False,
                                run_mlp=True,
                            )
                        else:
                            vlm_router_x, vlm_frozen_hidden = compute_frozen_prefix_layer(
                                layer_idx,
                                vlm_frozen_hidden,
                                attention_mask,
                                position_ids,
                                adarms_cond,
                                self.paligemma,
                                collect_mlp_input=layer_idx in vlm_idx,
                                run_mlp=layer_idx < vlm_fork_hi,
                            )
                if frozen_routing and fork_lo <= layer_idx <= fork_hi:
                    with torch.no_grad():
                        if layer_idx == fork_lo:
                            # Streams are identical before the fork layer's MLP: the
                            # live mlp-input IS the routing feature here (router_x
                            # stays None), and the frozen stream is initialized from
                            # this layer's plain-MLP output.
                            if fork_lo != fork_hi:
                                _, frozen_hidden = compute_frozen_suffix_layer(
                                    layer_idx,
                                    inputs_embeds[0],
                                    inputs_embeds[1],
                                    attention_mask,
                                    position_ids,
                                    adarms_cond,
                                    self.paligemma,
                                    self.gemma_expert,
                                    collect_mlp_input=False,
                                    run_mlp=True,
                                )
                        else:
                            router_x, frozen_hidden = compute_frozen_suffix_layer(
                                layer_idx,
                                inputs_embeds[0],
                                frozen_hidden,
                                attention_mask,
                                position_ids,
                                adarms_cond,
                                self.paligemma,
                                self.gemma_expert,
                                collect_mlp_input=layer_idx in mem_idx,
                                run_mlp=layer_idx < fork_hi,
                            )
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        context_fn=checkpoint_recompute_context_fn,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        task_emb=task_emb,
                        task_ids=task_ids,
                        router_x=router_x,
                        vlm_router_x=vlm_router_x,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        task_emb=task_emb,
                        task_ids=task_ids,
                        router_x=router_x,
                        vlm_router_x=vlm_router_x,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        task_emb=None,
        task_ids=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        self.paligemma_with_expert.set_vlm_token_mask(
            masks, instr_len=_vlm_instr_len_from_tokens(tokens),
            img_active=torch.stack(img_masks, dim=1) if img_masks else None,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                task_emb=task_emb,
                task_ids=task_ids,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        task_emb=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        self.paligemma_with_expert.set_vlm_token_mask(
            masks, instr_len=_vlm_instr_len_from_tokens(tokens),
            img_active=torch.stack(img_masks, dim=1) if img_masks else None,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            task_emb=task_emb,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    task_emb=task_emb,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        task_emb=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            task_emb=task_emb,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Attach memory layers to the action expert before loading any pretrained
        # weights, so memory parameters appear in the state_dict. PI05 does not
        # attach memory to the VLM backbone — only the action expert.
        if (
            getattr(self.config, "memory_layers", False)
            or getattr(self.config.memory_layer, "enabled", False)
        ) and not getattr(self.config, "pretrained_path", None):
            self.model.paligemma_with_expert.attach_memory_to_expert(self.config.memory_layer)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Optional language-conditioned query projection: cache sentence-transformers task
        # embeddings keyed by task name.
        self.task_embedding_cache = None
        if getattr(self.config.memory_layer, "lang_to_query", False):
            embedding_model = getattr(self.config.memory_layer, "embedding_model", "all-MiniLM-L6-v2")
            self.task_embedding_cache = TaskEmbeddingCache(model_name=embedding_model, device="cpu")

        # Paligemma tokenizer used by `get_task_embeddings_from_tokens` to decode the
        # language tokens back to text. Loaded lazily once per process, not per chunk.
        self._paligemma_tokenizer = None

        self.model.to(config.device)

        # Non-strict: on the from_pretrained path memory is attached later in
        # post_load_setup, which re-applies the freeze strictly.
        self._apply_train_memory_only(strict=not getattr(self.config, "pretrained_path", None))

        self.reset()

    def post_load_setup(self) -> None:
        """Hook called after loading pretrained weights.

        Re-attaches memory layers in case the loaded weights changed module
        structure (e.g., loading a non-memory checkpoint into a memory-enabled
        config, or vice versa).
        """
        if (
            getattr(self.config, "memory_layers", False)
            or getattr(self.config.memory_layer, "enabled", False)
        ):
            self.model.paligemma_with_expert.attach_memory_to_expert(self.config.memory_layer)
        self._apply_train_memory_only()

    def _apply_train_memory_only(self, strict: bool = True) -> None:
        """Freeze every parameter except the attached memory modules.

        Memory modules are identified by the ``.mlp.mem.`` name segment
        (``MLPPlusMemory`` wraps the expert MLP and registers the
        ``HashingMemoryLite`` as ``mem``); this keeps values, keys, query
        projections/FiLM and gating trainable while the whole backbone is
        frozen. Called from ``__init__`` (fresh attach) and ``post_load_setup``
        (attach after loading a checkpoint), so it holds for every load path.

        With ``strict=False`` (the ``__init__`` call on the from_pretrained
        path, where memory attaches later in ``post_load_setup``) a model with
        no memory params is left untouched instead of raising.

        ``train_router_only`` narrows the trainable set further to the memory
        ROUTER (keys + query projection/FiLM): values stay at init (slot_up is
        zero-init so the memory output — and hence the MSE gradient on the
        routing path — is ~0), giving a pure routing-loss warm-up phase.
        Supersedes ``train_memory_only`` when both are set.
        """
        router_only = getattr(self.config, "train_router_only", False)
        mem_only = getattr(self.config, "train_memory_only", False)
        freeze_router = getattr(self.config, "freeze_memory_router", False)
        if not (router_only or mem_only):
            return

        def is_router(name: str) -> bool:
            return ".mlp.mem.keys" in name or ".mlp.mem.query_proj." in name

        def trainable(name: str) -> bool:
            if router_only:
                return is_router(name)
            if ".mlp.mem." not in name:
                return False
            if freeze_router and is_router(name):
                return False
            return True

        names = [n for n, _ in self.named_parameters()]
        n_train = sum(trainable(n) for n in names)
        if n_train == 0:
            if strict:
                raise ValueError(
                    "train_memory_only/train_router_only set but no memory parameters "
                    "found - enable policy.memory_layers/memory_layer.enabled so memory "
                    "is attached."
                )
            return  # memory not attached yet; post_load_setup applies the freeze
        for name, param in self.named_parameters():
            param.requires_grad = trainable(name)
        mode = (
            "train_router_only" if router_only
            else "train_memory_only+freeze_memory_router" if freeze_router
            else "train_memory_only"
        )
        print(f"{mode}: {n_train} param tensors trainable, {len(names) - n_train} frozen")

    def precompute_task_embeddings(self, dataset_meta) -> None:
        if self.task_embedding_cache is not None:
            self.task_embedding_cache.precompute_from_metadata(dataset_meta)

    def get_task_embeddings(self, task_names: list[str]) -> torch.Tensor | None:
        if self.task_embedding_cache is None:
            return None
        return self.task_embedding_cache.get_by_indices(task_names)

    def get_task_embeddings_from_tokens(self, lang_tokens: torch.Tensor) -> torch.Tensor | None:
        """Fall back path used at inference: decode tokens to text and embed.

        Returns None if no task embedding cache is configured (memory disabled
        or no language-conditioned query). The caller treats None as "skip".

        Errors during tokenizer/encoder use are logged and re-raised — failing
        silently here was a real footgun: it produced a model that runs but
        ignores language conditioning entirely, which is indistinguishable from
        "memory disabled" at the call site.
        """
        if self.task_embedding_cache is None:
            return None
        if self._paligemma_tokenizer is None:
            from transformers import AutoTokenizer

            self._paligemma_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        try:
            texts = self._paligemma_tokenizer.batch_decode(lang_tokens, skip_special_tokens=True)
            texts = [t.strip() for t in texts]
            return self.task_embedding_cache.get_by_indices(texts)
        except Exception:
            logging.exception(
                "Failed to compute task embeddings from language tokens; "
                "memory routing would fall back to lang_emb=None."
            )
            raise

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # If the checkpoint already contains memory parameters, attach memory
            # before loading so the keys line up. Otherwise we'll attach after load
            # (handled inside __init__ via post_load_setup).
            checkpoint_has_memory = any(
                ".mlp.mem." in k or ".mlp.mlp." in k for k in original_state_dict.keys()
            )
            want_memory = (
                getattr(config, "memory_layers", False)
                or getattr(config.memory_layer, "enabled", False)
            )
            if checkpoint_has_memory and want_memory:
                model.model.paligemma_with_expert.attach_memory_to_expert(config.memory_layer)

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # When memory is enabled, the checkpoint may or may not contain memory
            # parameters depending on whether it was trained with memory. Allow
            # missing keys for the memory submodules so we can pretrain on top of
            # a non-memory base checkpoint.
            memory_enabled = (
                getattr(config, "memory_layers", False)
                or getattr(config.memory_layer, "enabled", False)
            )
            load_strict = strict and not memory_enabled

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=load_strict)

            # Filter out missing memory keys if memory is freshly attached
            if memory_enabled and missing_keys:
                memory_missing = [k for k in missing_keys if ".mlp.mem." in k or ".mlp.mlp." in k]
                missing_keys = [k for k in missing_keys if k not in memory_missing]
                if memory_missing:
                    print(
                        f"  ✓ {len(memory_missing)} memory param keys initialized from scratch "
                        "(checkpoint has no memory weights)"
                    )

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

            # Re-align memory module dtype/device after load: load_state_dict can
            # cast value params to match the checkpoint's dtype, but they should
            # stay in float32 for stable gradients.
            try:
                model.post_load_setup()
            except Exception:
                pass

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        # post_load_setup (which applies the train_memory_only freeze) runs inside
        # the try/except above; re-apply here so a swallowed load warning can never
        # leave the freeze silently unapplied.
        model._apply_train_memory_only()

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        if getattr(self.config, "memory_layers", False) or getattr(self.config.memory_layer, "enabled", False):
            mem_vals, others = split_memory_params(self)
            if len(mem_vals) == 0:
                return self.parameters()
            # Respect freezing (train_memory_only / train_router_only /
            # freeze_vision_encoder): frozen params get no grads anyway, but keeping
            # them out of the optimizer avoids allocating groups over dead params.
            others = [p for p in others if p.requires_grad]
            mem_vals = [p for p in mem_vals if p.requires_grad]
            if len(mem_vals) == 0:  # router warm-up: values frozen at init
                return [
                    {
                        "params": others,
                        "lr": self.config.optimizer_lr,
                        "weight_decay": self.config.optimizer_weight_decay,
                    }
                ]
            return [
                {
                    "params": others,
                    "lr": self.config.optimizer_lr,
                    "weight_decay": self.config.optimizer_weight_decay,
                },
                {
                    "params": mem_vals,
                    "lr": getattr(self.config.memory_layer, "memory_lr", 1e-3),
                    "weight_decay": getattr(self.config.memory_layer, "memory_weight_decay", 0.0),
                },
            ]
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        task_emb = self.get_task_embeddings_from_tokens(tokens)
        if task_emb is not None:
            task_emb = task_emb.to(device=tokens.device)

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, task_emb=task_emb, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        task_emb: Tensor | None = None,
        task_ids: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
            task_emb: Optional language embedding tensor for memory query conditioning.
            task_ids: Optional task index tensor used by memory contrastive/routing losses.
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        if task_ids is None and "task_index" in batch:
            task_ids = batch["task_index"]
            if task_ids is not None:
                task_ids = task_ids.to(device=images[0].device if images else tokens.device)

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            task_emb=task_emb,
            task_ids=task_ids,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["mse_loss"] = loss.item()

            # Aggregate optional memory regularizers (contrastive, routing) and
            # log per-layer + aggregate slot usage diagnostics when enabled.
            if (
                getattr(self.config, "memory_layers", False)
                or getattr(self.config.memory_layer, "enabled", False)
            ):
                try:
                    loss = aggregate_memory_losses(
                        self,
                        loss,
                        self.config.memory_layer,
                        loss_dict,
                    )
                except Exception:
                    pass

            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
