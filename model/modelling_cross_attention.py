"""
This code block is adapted from:
- Repository: OPT model in Huggingface Transformers
- Author: Huggingface
- Link: https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py
"""
# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MPT model."""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPTextModel,
    RobertaModel,
)

from transformers.utils import logging
logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class MPTConfig(PretrainedConfig):

    def __init__(self, args, opt_config, **kwargs):
        super().__init__(
            pad_token_id=opt_config.pad_token_id,
            bos_token_id=opt_config.bos_token_id,
            eos_token_id=opt_config.eos_token_id,
            **kwargs,
        )
        # MPT configuration
        self.neighbor_layer_wise = args.neighbor_layer_wise
        self.neighbor_mode = args.neighbor_mode
        self.peft_type = args.peft_type
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout

        # OPT configuration
        self.vocab_size = opt_config.vocab_size
        self.max_position_embeddings = opt_config.max_position_embeddings
        self.num_attention_heads = opt_config.num_attention_heads
        self.word_embed_proj_dim = opt_config.word_embed_proj_dim
        self.ffn_dim = opt_config.ffn_dim
        self.hidden_size = opt_config.hidden_size
        self.num_hidden_layers = opt_config.num_hidden_layers
        self.dropout = opt_config.dropout
        self.attention_dropout = opt_config.attention_dropout
        self.activation_function = opt_config.activation_function
        self.init_std = opt_config.init_std
        self.layerdrop = opt_config.layerdrop
        self.use_cache = opt_config.use_cache
        self.do_layer_norm_before = opt_config.do_layer_norm_before
        # We keep these variables at `True` for backward compatibility.
        self.enable_bias = opt_config.enable_bias
        self.layer_norm_elementwise_affine = opt_config.layer_norm_elementwise_affine

        # Note that the only purpose of `_remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        self._remove_final_layer_norm = opt_config._remove_final_layer_norm


class MPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class MPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, cross_attention):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        bias = config.enable_bias

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        self.cross_attention = cross_attention
        self.peft_type = config.peft_type
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        neighbor_embeds: Optional[torch.Tensor] = None,
        neighbor_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if self.cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(neighbor_embeds), -1, bsz)
            value_states = self._shape(self.v_proj(neighbor_embeds), -1, bsz)
            attention_mask = neighbor_attention_mask
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, None


class MPTDecoderLayer(nn.Module):
    def __init__(self, config, cross_attention=False):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MPTAttention(config, cross_attention)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        self.cross_attention = cross_attention
        self.peft_type = config.peft_type
        if self.cross_attention and self.peft_type == "flamingo":
            self.tanh_layer1 = nn.Tanh()
            self.tanh_layer2 = nn.Tanh()
            self.gating1 = nn.Parameter(torch.tensor(0.0))
            self.gating2 = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        neighbor_embeds: Optional[torch.FloatTensor] = None,
        neighbor_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            neighbor_embeds=neighbor_embeds,
            neighbor_attention_mask=neighbor_attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.cross_attention and self.peft_type == "flamingo":
            hidden_states = residual + self.tanh_layer1(self.gating1) * hidden_states
        else:
            hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.cross_attention and self.peft_type == "flamingo":
            hidden_states = (residual + self.tanh_layer2(self.gating2) * hidden_states).view(hidden_states_shape)
        else:
            hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MPTDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (MPTDecoder)):
            module.gradient_checkpointing = value


class MPTDecoder(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = MPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.cross_attention = (config.neighbor_mode == "cross_attention")
        self.neighbor_layer_wise = config.neighbor_layer_wise
        self.peft_type = config.peft_type

        self.layers = nn.ModuleList()
        self.neighbor_layers = nn.ModuleList()
        for l in range(config.num_hidden_layers):
            self.layers.append(MPTDecoderLayer(config))
            if self.cross_attention and (l + 1) % self.neighbor_layer_wise == 0:
                self.neighbor_layers.append(MPTDecoderLayer(config, cross_attention=True))

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        neighbor_embeds: Optional[torch.FloatTensor] = None,
        neighbor_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Args:
            input_ids : token ids of input text
            attention_mask : attention_mask of input text
            head_mask : mask selected heads of the attention modules
            past_key_values : previous key values of the decoder
            inputs_embeds : embeddings of input text
            neighbor_embeds : embeddings of neighbor text/images
            neighbor_attention_mask : attention mask of neighbor text/images
            use_cache : whether to use cache
            output_attentions : whether to output attentions
            output_hidden_states : whether to output hidden states
            return_dict : whether to return a dict
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        if neighbor_attention_mask is not None:
            neighbor_attention_mask = _expand_mask(neighbor_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                if self.cross_attention and (idx + 1) % self.neighbor_layer_wise == 0:
                    hidden_states = layer_outputs[0]
                    neighbor_idx = (idx + 1) // self.neighbor_layer_wise - 1
                    layer_outputs = self.neighbor_layers[neighbor_idx](
                        hidden_states,
                        attention_mask=causal_attention_mask,
                        neighbor_embeds=neighbor_embeds,
                        neighbor_attention_mask=neighbor_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MPTModel(MPTPreTrainedModel):
    def __init__(self, config: MPTConfig):
        super().__init__(config)
        self.decoder = MPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        neighbor_embeds: Optional[torch.FloatTensor] = None,
        neighbor_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            neighbor_embeds=neighbor_embeds,
            neighbor_attention_mask=neighbor_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


def reset_peft_parameters(model):
    for n, p in model.named_parameters():
        if "lora_A" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if "lora_B" in n:
            nn.init.zeros_(p)
        if "adapter" in n:
            identity = torch.eye(p.size(0), p.size(1))
            # Add small random noise
            noise = torch.randn(p.size(0), p.size(1)) * 0.01
            p = identity + noise

def mark_only_peft_as_trainable(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, MPTDecoderLayer) and m.cross_attention == True:
            for n, p in m.named_parameters():
                p.requires_grad = True

class MPTForCausalLM(MPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        if config.peft_type != 'none':
            reset_peft_parameters(self.model)
            mark_only_peft_as_trainable(self.model)

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        neighbor_embeds: Optional[torch.FloatTensor] = None,
        neighbor_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            input_ids : token ids of input text
            attention_mask : attention_mask of input text
            head_mask : mask selected heads of the attention modules
            past_key_values : previous key values of the decoder
            inputs_embeds : embeddings of input text
            labels : token ids of output text
            neighbor_embeds : embeddings of neighbor text/images
            neighbor_attention_mask : attention mask of neighbor text/images
            use_cache : whether to use cache
            output_attentions : whether to output attentions
            output_hidden_states : whether to output hidden states
            return_dict : whether to return a dict
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            neighbor_embeds=neighbor_embeds,
            neighbor_attention_mask=neighbor_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class TextPooler(nn.Module):
    """
    Pool the hidden state corresponding to the first token.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossAttentionModel(nn.Module):
    """
    CrossAttentionModel is a wrapper around a pretrained language model.
    It supports the decoder-only models (e.g., OPT).)
    """
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.context = args.context
        self.neighbor_mode = args.neighbor_mode
        self.n_text_tokens = args.n_text_tokens
        self.n_visual_tokens = args.n_visual_tokens
        self.tokenizer = tokenizer

        self.initialize_lm(args)
        self.input_embeddings = self.lm.get_input_embeddings()

        self.text_model = None
        if self.context != "section_only":
            # Text model to encode text neighbors
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            if "clip" in args.text_model:
                self.text_model = CLIPTextModel.from_pretrained(args.text_model)
            else:
                self.text_model = RobertaModel.from_pretrained(args.text_model)
                self.text_pooler = TextPooler(self.text_model.config)
            self.text_embeddings = nn.Linear(self.text_model.config.hidden_size, embedding_dim)
            self.text_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors
            # Freeze the text model
            self.text_model.eval()
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False

        self.visual_model = None
        if self.context in ("section_all", "all"):
            # Vision model to encode image neighbors
            embedding_dim = self.input_embeddings.embedding_dim * args.n_visual_tokens
            self.visual_model = CLIPVisionModel.from_pretrained(args.visual_model)
            self.visual_embeddings = nn.Linear(self.visual_model.config.hidden_size, embedding_dim)
            self.visual_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors
            # Freeze the vision model
            self.visual_model.eval()
            for param in self.visual_model.parameters():
                param.requires_grad = False

        # Freeze the base LM if needed
        if self.args.freeze_lm:
            print("Freezing the LM.")
            self.lm.eval()
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

    def initialize_lm(self, args):
        # Initialize the LM using the pretrained model except cross-attention layers 
        opt_config = AutoConfig.from_pretrained(args.model_name_or_path)
        opt_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=opt_config)

        mpt_config = MPTConfig(args, opt_config)
        mpt_model = MPTForCausalLM(mpt_config)

        # Copy embeddings
        mpt_model.model.decoder.embed_tokens.load_state_dict(opt_model.model.decoder.embed_tokens.state_dict())
        mpt_model.model.decoder.embed_positions.load_state_dict(opt_model.model.decoder.embed_positions.state_dict())
        if mpt_config.word_embed_proj_dim != mpt_config.hidden_size:
            mpt_model.model.decoder.project_out.load_state_dict(opt_model.model.decoder.project_out.state_dict())
            mpt_model.model.decoder.project_in.load_state_dict(opt_model.model.decoder.project_in.state_dict())
        if mpt_config.do_layer_norm_before and not mpt_config._remove_final_layer_norm:
            mpt_model.model.decoder.final_layer_norm.load_state_dict(opt_model.model.decoder.final_layer_norm.state_dict())

        # Copy self-attention layers
        for idx in range(opt_config.num_hidden_layers):
            missing_keys, unexpected_keys = mpt_model.model.decoder.layers[idx].load_state_dict(opt_model.model.decoder.layers[idx].state_dict(), strict=False)
            print(f'{idx}th layer missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}')

        # Copy lm_head
        mpt_model.lm_head.load_state_dict(opt_model.lm_head.state_dict())

        self.lm = mpt_model

    def get_text_embs(self, input_ids, attention_mask, pos_ids=None):
        """
        Get the text embeddings from the text model.
        Args:
            input_ids: token ids of text neighbors (batch_size, neighbor_num, seq_len)
            attention_mask: attention mask of text neighbors (batch_size, neighbor_num, seq_len)
            pos_ids: position ids of text neighbors (batch_size, neighbor_num, seq_len)
        Returns:
            text_embs: text embeddings of text neighbors (batch_size, neighbor_num, n_text_tokens, hidden_dim)
        """
        batch_size, neighbor_num, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        if "clip" in self.args.text_model:
            encoder_outputs = outputs.pooler_output
        else:
            encoder_outputs = self.text_pooler(outputs.last_hidden_state)
        text_embs = self.text_embeddings(encoder_outputs)

        if pos_ids is not None:
            pos_ids = pos_ids.reshape(-1)
            text_embs = text_embs + self.text_position_embeddings(pos_ids)

        text_embs = text_embs.reshape(text_embs.shape[0], self.n_text_tokens, -1)
        return text_embs.reshape(batch_size, neighbor_num, self.n_text_tokens, -1)

    def get_visual_embs(self, pixel_values, pos_ids=None):
        """
        Get the visual embeddings from the vision model.
        Args:
            pixel_values: pixel values of image neighbors (batch_size, neighbor_num, pixel, width, height)
            pos_ids: position ids of image neighbors (batch_size, neighbor_num)
        Returns:
            visual_embs: visual embeddings of image neighbors (batch_size, neighbor_num, n_visual_tokens, hidden_dim)
        """
        batch_size, neighbor_num, pixel, width, height = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, pixel, width, height)

        outputs = self.visual_model(pixel_values)
        encoder_outputs = outputs.pooler_output
        visual_embs = self.visual_embeddings(encoder_outputs)

        if pos_ids is not None:
            pos_ids = pos_ids.reshape(-1)
            visual_embs = visual_embs + self.visual_position_embeddings(pos_ids)

        visual_embs = visual_embs.reshape(visual_embs.shape[0], self.n_visual_tokens, -1)
        return visual_embs.reshape(batch_size, neighbor_num, self.n_visual_tokens, -1)

    def train(self, mode=True):
        super(CrossAttentionModel, self).train(mode=mode)
        if self.args.freeze_lm:
            self.lm.eval()
        if self.text_model is not None:
            self.text_model.eval()
        if self.visual_model is not None:
            self.visual_model.eval()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        images=None,
        image_positions=None,
        neighbor_input_ids=None,
        neighbor_attention_mask=None,
        neighbor_pos_ids=None,
        text_locations=None,
        neighbor_images=None,
        neighbor_images_pos_ids=None,
        image_locations=None,
    ):
        """
        Args:
            input_ids: token ids of input text (batch_size, seq_len)
            attention_mask: attention_mask of input text (batch_size, seq_len)
            labels: token ids of labels (batch_size, seq_len)
            images: neighbor image features (batch_size, image_num, pixel, width, height)
            image_positions: neighbor image locations (batch_size, image_num)
            neighbor_input_ids: token ids of neighbor text (batch_size, text_num, seq_len)
            neighbor_attention_mask: attention mask of neighbor text (batch_size, text_num, seq_len)
            neighbor_pos_ids: position ids of neighbor text (batch_size, text_num, seq_len)
            text_locations: locations of text embeddings (batch_size, text_num)
            neighbor_images: neighbor image features (batch_size, image_num, pixel, width, height)
            neighbor_images_pos_ids: position ids of neighbor images (batch_size, image_num)
            image_locations: locations of image embeddings (batch_size, image_num)
        """
        if self.neighbor_mode == "raw" or self.context == "section_only":
            # For sanity check: run the pure OPT model
            neighbor_embeds = None
            neighbor_attention_mask = None
        elif self.neighbor_mode == "cross_attention" and self.context == "text_only":
            # Text neighbors only; need to compute text embeddings
            batch_size, neighbor_num, seq_len = neighbor_input_ids.shape
            neighbor_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
            neighbor_embeds = neighbor_embeds.reshape(batch_size, neighbor_num * self.n_text_tokens, -1)
            neighbor_attention_mask = neighbor_pos_ids > 0
            neighbor_attention_mask = torch.repeat_interleave(neighbor_attention_mask, repeats=self.n_text_tokens, dim=1)

        elif self.neighbor_mode == "cross_attention" and self.context in ("section_all", "all"):
            # Text and image neighbors; need to compute text and image embeddings
            text_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
            batch_size, text_neighbor_num, n_tokens, hidden_dim = text_embeds.shape
            text_attention_mask = neighbor_pos_ids > 0
            text_attention_mask = text_attention_mask.unsqueeze(-1).expand(-1, -1, self.n_text_tokens)

            visual_embeds = self.get_visual_embs(neighbor_images, neighbor_images_pos_ids)
            batch_size, visual_neighbor_num, n_tokens, hidden_dim = visual_embeds.shape
            visual_attention_mask = neighbor_images_pos_ids > 0
            visual_attention_mask = visual_attention_mask.unsqueeze(-1).expand(-1, -1, self.n_visual_tokens)

            # Interleave text and image neighbors
            batch_idx = torch.arange(batch_size)[:, None]
            total_neighbor_num = text_neighbor_num + visual_neighbor_num
            neighbor_embeds = torch.zeros((batch_size, total_neighbor_num, n_tokens, hidden_dim)).to(neighbor_input_ids.device)
            neighbor_embeds[batch_idx, text_locations] = text_embeds
            neighbor_embeds[batch_idx, image_locations] = visual_embeds
            neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, hidden_dim)

            # Interleave text and image attention masks
            neighbor_attention_mask = torch.zeros((batch_size, total_neighbor_num, n_tokens)).bool().to(neighbor_attention_mask.device)
            neighbor_attention_mask[batch_idx, text_locations] = text_attention_mask
            neighbor_attention_mask[batch_idx, image_locations] = visual_attention_mask
            neighbor_attention_mask = neighbor_attention_mask.reshape(batch_size, -1)
        else:
            raise ValueError(f"Neighbor mode: {self.neighbor_mode} and context: {self.context} are not supported.")

        output = self.lm(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels,
                         neighbor_embeds=neighbor_embeds,
                         neighbor_attention_mask=neighbor_attention_mask)

        return output
