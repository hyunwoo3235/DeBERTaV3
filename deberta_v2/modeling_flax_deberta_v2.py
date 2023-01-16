# coding=utf-8
# Copyright 2021 Microsoft and The HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers.modeling_flax_outputs import (
    FlaxMaskedLMOutput,
    FlaxBaseModelOutput,
)
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import logging

from .configuration_deberta_v2 import DebertaV2Config

logger = logging.get_logger(__name__)

remat = nn_partitioning.remat


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        input_ids: jnp.ndarray
        padding_idx: int
    Returns: jnp.ndarray
    """
    mask = (input_ids != padding_idx).astype("i4")

    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    return incremental_indices.astype("i4") + padding_idx


class FlaxContextPooler(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.pooler_hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.pooler_dropout)

    def __call__(self, hidden_states, deterministic: bool = True):
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token, deterministic=deterministic)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output


class FlaxDebertaV2SelfOutput(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlaxDebertaV2Attention(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxDisentangledSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxDebertaV2SelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        deterministic=True,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )

        if output_attentions:
            self_outputs, att_matrix = self_outputs
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(
            self_outputs, query_states, deterministic=deterministic
        )

        if output_attentions:
            return attention_output, att_matrix

        return attention_output


class FlaxDebertaV2Intermediate(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.intermediate_act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FlaxDebertaV2Output(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxDebertaV2Layer(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxDebertaV2Attention(self.config, dtype=self.dtype)
        self.intermediate = FlaxDebertaV2Intermediate(self.config, dtype=self.dtype)
        self.output = FlaxDebertaV2Output(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions: bool = False,
        deterministic: bool = True,
    ):
        # Self Attention
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            deterministic=deterministic,
        )
        if output_attentions:
            attention_out, att_matrix = attention_output

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if output_attentions:
            return layer_output, att_matrix
        return (layer_output,)


class FlaxDebertaV2ConvLayer(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        pass

    def __call__(
        self, hidden_states, residual_states, input_mask, deterministic: bool = True
    ):
        raise NotImplementedError


class FlaxDebertaV2LayerCollection(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxDebertaV2Layer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

        self.conv = (
            FlaxDebertaV2ConvLayer(self.config, dtype=self.dtype)
            if getattr(self.config, "conv_kernel_size", 0) > 0
            else None
        )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        query_states: Optional[jnp.ndarray] = None,
        relative_pos: Optional[jnp.ndarray] = None,
        rel_embeddings: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_dict: bool = True,
    ):
        if attention_mask.ndim <= 2:
            input_mask = attention_mask
        else:
            input_mask = (jnp.sum(attention_mask, axis=-2) > 0).astype("i4")

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        next_kv = hidden_states
        output_states = next_kv
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (output_states,)

            layer_outputs = layer(
                next_kv,
                attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            output_states = layer_outputs[0]

            if i == 0 and self.conv is not None:
                output_states = self.conv(
                    hidden_states,
                    output_states,
                    input_mask,
                    deterministic=deterministic,
                )

            next_kv = output_states

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (output_states,)

        if not return_dict:
            return tuple(
                v
                for v in [output_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxDebertaV2Encoder(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer = FlaxDebertaV2LayerCollection(
            self.config,
            dtype=self.dtype,
        )
        self.relative_attention = getattr(self.config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(
                self.config, "max_relative_positions", -1
            )
            if self.max_relative_positions < 1:
                self.max_relative_positions = self.config.max_position_embeddings

            self.position_buckets = getattr(self.config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = self.param(
                "rel_embeddings.weight",
                jax.nn.initializers.normal(self.config.initializer_range),
                (pos_ebd_size, self.config.hidden_size),
                self.dtype,
            )

        self.norm_rel_ebd = [
            x.strip()
            for x in getattr(self.config, "norm_rel_ebd", "none").lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(
                epsilon=self.config.layer_norm_eps, dtype=self.dtype
            )

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.ndim <= 2:
            extended_attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
            attention_mask = extended_attention_mask * jnp.expand_dims(
                jnp.squeeze(extended_attention_mask, axis=-2), axis=-1
            )
            attention_mask = attention_mask.astype("i4")
        elif attention_mask.ndim == 3:
            attention_mask = jnp.expand_dims(attention_mask, axis=1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = (
                query_states.shape[-2]
                if query_states is not None
                else hidden_states.shape[-2]
            )
            relative_pos = build_relative_position(
                q,
                hidden_states.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        query_states: Optional[jnp.ndarray] = None,
        relative_pos: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_dict: bool = True,
    ):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        rel_embeddings = self.get_rel_embedding()

        return self.layer(
            hidden_states,
            attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )


class FlaxDebertaV2Embeddings(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding_size = getattr(
            self.config, "embedding_size", self.config.hidden_size
        )
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
        )

        self.position_biased_input = getattr(self.config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embed(
                self.config.max_position_embeddings,
                self.config.hidden_size,
                embedding_init=jax.nn.initializers.normal(
                    stddev=self.config.initializer_range
                ),
                dtype=self.dtype,
            )

        if self.config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embed(
                self.config.type_vocab_size,
                self.config.hidden_size,
                embedding_init=jax.nn.initializers.normal(
                    stddev=self.config.initializer_range
                ),
                dtype=self.dtype,
            )

        if self.embedding_size != self.config.hidden_size:
            self.embed_proj = nn.Dense(
                self.config.hidden_size, use_bias=False, dtype=False
            )
        self.LayerNorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        mask=None,
        inputs_embeds=None,
        deterministic: bool = True,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if position_ids is None:
            position_ids = jnp.expand_dims(jnp.arange(0, input_shape[-1]), axis=0)

        if token_type_ids is None:
            token_type_ids = jnp.zeros(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids.astype("i4"))

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.astype("i4"))
        else:
            position_embeddings = jnp.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(
                token_type_ids.astype("i4")
            )
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.ndim != embeddings.ndim:
                if mask.ndim == 4:
                    mask = jnp.squeeze(jnp.squeeze(mask, axis=1), axis=1)
                mask = jnp.expand_dims(mask, axis=2)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = jnp.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = jnp.where(
        (relative_pos < mid) & (relative_pos > -mid),
        mid - 1,
        jnp.abs(relative_pos),
    )
    log_pos = (
        jnp.ceil(jnp.log(abs_pos / mid) / jnp.log((max_position - 1) / mid) * (mid - 1))
        + mid
    )
    bucket_pos = jnp.where(abs_pos <= mid, abs_pos, log_pos * sign)
    return bucket_pos.astype("i4")


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    q_ids = jnp.arange(0, query_size)
    k_ids = jnp.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - jnp.tile(
        jnp.expand_dims(k_ids, axis=0), (q_ids.shape[0], 1)
    )
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.astype("i4")
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = jnp.expand_dims(rel_pos_ids, axis=0)
    return rel_pos_ids


def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    shapes = [
        query_layer.shape[0],
        query_layer.shape[1],
        query_layer.shape[2],
        relative_pos.shape[-1],
    ]
    return jnp.broadcast_to(c2p_pos, shapes)


def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    shapes = [
        query_layer.shape[0],
        query_layer.shape[1],
        key_layer.shape[-2],
        key_layer.shape[-2],
    ]
    return jnp.broadcast_to(c2p_pos, shapes)


def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    shapes = p2c_att.shape[:2] + [pos_index.shape[-2]] + key_layer.shape[-2]
    return jnp.broadcast_to(pos_index, shapes)


class FlaxDisentangledSelfAttention(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_attention_heads = self.config.num_attention_heads
        _attention_head_size = (
            self.config.hidden_size // self.config.num_attention_heads
        )
        self.attention_head_size = getattr(
            self.config, "attention_head_size", _attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_proj = nn.Dense(
            self.all_head_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key_proj = nn.Dense(
            self.all_head_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value_proj = nn.Dense(
            self.all_head_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.share_att_key = getattr(self.config, "share_att_key", False)
        self.pos_att_type = (
            self.config.pos_att_type if self.config.pos_att_type is not None else []
        )
        self.relative_attention = getattr(self.config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(self.config, "position_buckets", -1)
            self.max_relative_positions = getattr(
                self.config, "max_relative_positions", -1
            )
            if self.max_relative_positions < 1:
                self.max_relative_positions = self.config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Dense(
                        self.all_head_size,
                        dtype=self.dtype,
                        kernel_init=jax.nn.initializers.normal(
                            self.config.initializer_range
                        ),
                    )
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Dense(
                        self.all_head_size,
                        dtype=self.dtype,
                        kernel_init=jax.nn.initializers.normal(
                            self.config.initializer_range
                        ),
                    )

        self.dropout = nn.Dropout(rate=self.config.attention_probs_dropout_prob)

    @staticmethod
    def transpose_for_scores(x, attention_heads):
        shape = x.shape[:-1] + (attention_heads, x.shape[-1] // attention_heads)
        x = jnp.reshape(x, shape)
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = jnp.reshape(x, (-1, x.shape[-2], x.shape[-1]))
        return x

    def __call__(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
        deterministic=True,
    ):
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        )

        rel_att = None
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = jnp.sqrt(query_layer.shape[-1] * scale_factor)
        attention_scores = (
            jnp.matmul(query_layer, jnp.transpose(key_layer, (0, 2, 1))) / scale
        )
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(
                rel_embeddings, deterministic=deterministic
            )
            rel_att = self.disentangled_att_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = jnp.reshape(
            attention_scores,
            (
                -1,
                self.num_attention_heads,
                attention_scores.shape[-2],
                attention_scores.shape[-1],
            ),
        )

        attention_probs = nn.softmax(attention_scores, -1)
        attention_probs = self.dropout(attention_probs, deterministic=deterministic)
        context_layer = jnp.matmul(
            jnp.reshape(
                attention_probs,
                (-1, attention_probs.shape[-2], attention_probs.shape[-1]),
            ),
            value_layer,
        )
        context_layer = jnp.transpose(
            jnp.reshape(
                context_layer,
                (
                    -1,
                    self.num_attention_heads,
                    context_layer.shape[-2],
                    context_layer.shape[-1],
                ),
            ),
            (0, 2, 1, 3),
        )

        new_context_layer_shape = context_layer.shape[:-2] + (-1,)
        context_layer = jnp.reshape(context_layer, new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        return context_layer

    def disentangled_att_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(
                q,
                key_layer.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.ndim == 2:
            relative_pos = jnp.expand_dims(relative_pos, axis=0)
            relative_pos = jnp.expand_dims(relative_pos, axis=0)
        elif relative_pos.ndim == 3:
            relative_pos = jnp.expand_dims(relative_pos, axis=1)
        elif relative_pos.ndim != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.ndim}"
            )

        att_span = self.pos_ebd_size

        rel_embeddings = jnp.expand_dims(
            rel_embeddings[
                self.pos_ebd_size - att_span : self.pos_ebd_size + att_span, :
            ],
            0,
        )
        if self.share_att_key:
            pos_query_layer = jnp.tile(
                self.transpose_for_scores(
                    self.query_proj(rel_embeddings), self.num_attention_heads
                ),
                (query_layer.shape[0] // self.num_attention_heads, 1, 1),
            )
            pos_key_layer = jnp.tile(
                self.transpose_for_scores(
                    self.key_proj(rel_embeddings), self.num_attention_heads
                ),
                (query_layer.shape[0] // self.num_attention_heads, 1, 1),
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = jnp.tile(
                    self.transpose_for_scores(
                        self.pos_key_proj(rel_embeddings), self.num_attention_heads
                    ),
                    (query_layer.shape[0] // self.num_attention_heads, 1, 1),
                )
            if "p2c" in self.pos_att_type:
                pos_query_layer = jnp.tile(
                    self.transpose_for_scores(
                        self.pos_query_proj(rel_embeddings), self.num_attention_heads
                    ),
                    (query_layer.shape[0] // self.num_attention_heads, 1, 1),
                )

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = jnp.sqrt(pos_key_layer.shape[-1] * scale_factor)
            c2p_att = jnp.matmul(query_layer, jnp.transpose(pos_key_layer, (0, 2, 1)))
            c2p_pos = jnp.clip(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = jnp.take_along_axis(
                c2p_att,
                jnp.broadcast_to(
                    jnp.squeeze(c2p_pos, axis=0),
                    (
                        query_layer.shape[0],
                        query_layer.shape[1],
                        relative_pos.shape[-1],
                    ),
                ),
                axis=-1,
            )
            score += c2p_att / scale

        # position->content
        if "p2c" in self.pos_att_type:
            scale = jnp.sqrt(pos_query_layer.shape[-1] * scale_factor)
            if key_layer.shape[-2] != relative_pos.shape[-1]:
                r_pos = build_relative_position(
                    key_layer.shape[-2],
                    key_layer.shape[-2],
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
                r_pos = jnp.expand_dims(r_pos, axis=0)
            else:
                r_pos = relative_pos

            p2c_pos = jnp.clip(-r_pos + att_span, 0, att_span * 2 - 1)

            p2c_att = jnp.matmul(key_layer, jnp.transpose(pos_key_layer, (0, 2, 1)))
            p2c_att = jnp.transpose(
                jnp.take_along_axis(
                    p2c_att,
                    jnp.broadcast_to(
                        jnp.squeeze(p2c_pos, 0),
                        (
                            query_layer.shape[0],
                            key_layer.shape[-2],
                            key_layer.shape[-2],
                        ),
                    ),
                    axis=-1,
                ),
                (0, 2, 1),
            )
            score += p2c_att / scale

        return score


class FlaxDebertaV2PredictionHeadTransform(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
            dtype=self.dtype,
        )
        if isinstance(self.config.hidden_act, str):
            self.transform_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.transform_act_fn = self.config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FlaxDebertaV2LMPredictionHead(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transform = FlaxDebertaV2PredictionHeadTransform(
            self.config, dtype=self.dtype
        )

        self.decoder = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
            dtype=self.dtype,
        )
        self.bias = self.param(
            "bias", jax.nn.initializers.zeros, (self.config.vocab_size,)
        )

    def __call__(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        hidden_states = hidden_states + self.bias
        return hidden_states


class FlaxDebertaV2OnlyMLMHead(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxDebertaV2LMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.predictions(hidden_states)
        return hidden_states


class FlaxDebertaV2DiscriminatorPredictions(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
            dtype=self.dtype,
        )
        if isinstance(self.config.hidden_act, str):
            self.transform_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.transform_act_fn = self.config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )

        self.classifier = nn.Dense(
            1,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        ctx_states = hidden_states[:, 0, :]
        hidden_states = self.LayerNorm(
            jnp.expand_dims(ctx_states, axis=-2) + hidden_states
        )
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)

        logits = self.classifier(hidden_states).squeeze(-1)
        return logits


class FlaxDebertaV2PreTrainedModel(FlaxPreTrainedModel):
    config_class = DebertaV2Config
    base_model_prefix = "deberta_v2"

    module_class: nn.Module = None

    def __init__(
        self,
        config: DebertaV2Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.ones_like(input_ids)
        position_ids = create_position_ids_from_input_ids(
            input_ids, self.config.pad_token_id
        )
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            return_dict=False,
        )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(
                input_ids, self.config.pad_token_id
            )

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            token_type_ids=jnp.array(token_type_ids, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            deterministic=not train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
        )

        return outputs


class FlaxDebertaV2Module(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxDebertaV2Embeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxDebertaV2Encoder(
            self.config,
            dtype=self.dtype,
        )
        self.z_steps = 0

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = jnp.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = jnp.zeros(input_shape, dtype="i4")

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
            deterministic=deterministic,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
            deterministic=deterministic,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            raise NotImplementedError

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[
                (1 if output_hidden_states else 2) :
            ]

        return FlaxBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states
            if output_hidden_states
            else None,
            attentions=encoder_outputs.attentions,
        )


class FlaxDebertaV2Model(FlaxDebertaV2PreTrainedModel):
    module_class = FlaxDebertaV2Module


class FlaxDebertaV2ForMaskedLMModule(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.deberta = FlaxDebertaV2Module(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
        )
        self.cls = FlaxDebertaV2OnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        inputs_embeds=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Model
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if not return_dict:
            return (prediction_scores,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxDebertaV2ForMaskedLM(FlaxDebertaV2PreTrainedModel):
    module_class = FlaxDebertaV2ForMaskedLMModule


class FlaxDebertaV2ForPreTrainingModule(nn.Module):
    config: DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.deberta = FlaxDebertaV2Module(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
        )
        self.mask_predictions = FlaxDebertaV2DiscriminatorPredictions(
            config=self.config, dtype=self.dtype
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        inputs_embeds=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Model
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        mask_prediction_scores = self.mask_predictions(sequence_output)

        if not return_dict:
            return (mask_prediction_scores,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=mask_prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxDebertaV2ForPreTraining(FlaxDebertaV2PreTrainedModel):
    module_class = FlaxDebertaV2ForPreTrainingModule
