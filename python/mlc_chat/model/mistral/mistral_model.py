"""
Implementation for Mistral architecture.
"""
import dataclasses
from typing import Any, Dict, Optional

import tvm
from tvm import relax as rx
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_chat import op as op_ext
from mlc_chat.support import logging
from mlc_chat.support import tensor_parallel as tp
from mlc_chat.support.config import ConfigBase
from mlc_chat.support.style import bold

logger = logging.getLogger(__name__)

from .vit_model import ViT, ViTConfig

@dataclasses.dataclass
class MistralConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Mistral model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    position_embedding_base: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    sliding_window_size: int = 4096
    prefill_chunk_size: int = 0
    attention_sink_size: int = 4
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size
        assert self.attention_sink_size >= 0
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %s (%d)",
                bold("prefill_chunk_size"),
                bold("sliding_window_size"),
                self.sliding_window_size,
            )
            self.prefill_chunk_size = self.sliding_window_size
        elif self.prefill_chunk_size > self.sliding_window_size:
            logger.info(
                "Overriding %s from %d to %d (%s)",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                self.sliding_window_size,
                bold("sliding_window_size"),
            )
            self.prefill_chunk_size = self.sliding_window_size


# pylint: disable=invalid-name,missing-docstring


class RotaryEmbedding(nn.Module):
    """Cache relative Rotary Embedding."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.position_embedding_base = config.position_embedding_base

    def forward(self, q: Tensor, k: Tensor, q_offset: tir.Var):
        def te_op(x: te.Tensor, offset: tir.Var):
            dtype = x.dtype

            def compute(b: tir.Var, s: tir.Var, h: tir.Var, d: tir.Var):
                head_dim = tir.const(self.head_dim, "int32")
                position_embedding_base = tir.const(self.position_embedding_base, "float32")
                freq = tir.power(
                    position_embedding_base,
                    (d * 2 % head_dim).astype("float32") / head_dim,
                )
                freq = (offset + s) / freq
                cos = tir.cos(freq).astype(dtype) * x[b, s, h, d]
                sin = tir.sin(freq).astype(dtype) * tir.if_then_else(
                    d < head_dim // 2,
                    -x[b, s, h, d + head_dim // 2],
                    x[b, s, h, d - head_dim // 2],
                )
                return cos + sin

            return te.compute(x.shape, compute, name="rotary")

        q_embed = op.tensor_expr_op(
            te_op,
            "rotary_embedding",
            args=[q, q_offset],
            attrs={"mlc.rotary_embedding_to_all_dims": True},
        )
        k_embed = op.tensor_expr_op(
            te_op, "rotary_embedding", args=[k, 0], attrs={"mlc.rotary_embedding_to_all_dims": True}
        )
        return q_embed, k_embed


class MistralMLP(nn.Module):
    """Same as in Llama architecture (LlamaFFN)."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


class MistralAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Same as LlamaAttention, but with sliding window attention using a rolling buffer cache."""

    def __init__(self, config: MistralConfig, rotary_embedding: RotaryEmbedding):
        self.rotary_embedding = rotary_embedding
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.sliding_window_size = config.sliding_window_size
        self.attention_sink_size = config.attention_sink_size
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)
        self.k_cache = RollingKVCacheWithSinks(
            self.sliding_window_size, [self.num_kv_heads, self.head_dim]
        )
        self.v_cache = RollingKVCacheWithSinks(
            self.sliding_window_size, [self.num_kv_heads, self.head_dim]
        )

    def interleave_kv(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        k_cur: Tensor,
        v_cur: Tensor,
        kv_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """Unrotate and concatenate currunt and cached k and v"""
        h_kv, d = self.num_kv_heads, self.head_dim
        kv_s, c, o = kv_seq_len, rolling_cache_len, cache_offset
        b = k_cur.shape[0]

        k_cached = op.reshape(self.k_cache.view(c), (b, c, h_kv, d))
        v_cached = op.reshape(self.v_cache.view(c), (b, c, h_kv, d))

        def _cache_unrotate(x_cached, rolling_cache_len, cache_offset):
            return te.compute(
                (b, kv_s, h_kv, d),
                lambda xb, xs, xh, xd: te.if_then_else(
                    xs < self.attention_sink_size,
                    x_cached[xb, xs, xh, xd],
                    te.if_then_else(
                        xs < rolling_cache_len - cache_offset + self.attention_sink_size,
                        x_cached[xb, xs + cache_offset - self.attention_sink_size, xh, xd],
                        x_cached[xb, xs + cache_offset - rolling_cache_len, xh, xd],
                    ),
                ),
                name="cache_unrotate_te",
            )

        def _cache_cur_concat(x_cached, x_cur, rolling_cache_len):
            return te.compute(
                (b, kv_s, h_kv, d),
                lambda xb, xs, xh, xd: te.if_then_else(
                    xs < rolling_cache_len,
                    x_cached[xb, xs, xh, xd],
                    x_cur[xb, xs - rolling_cache_len, xh, xd],
                ),
                name="cache_cur_concat_te",
            )

        k_cached = op.tensor_expr_op(
            _cache_unrotate,
            name_hint="te_cache_unrotate_key",
            args=[k_cached, c, o],
        )
        k = op.tensor_expr_op(
            _cache_cur_concat,
            name_hint="te_cache_cur_concat_key",
            args=[k_cached, k_cur, c],
        )

        v_cached = op.tensor_expr_op(
            _cache_unrotate,
            name_hint="te_cache_unrotate_value",
            args=[v_cached, c, o],
        )
        v = op.tensor_expr_op(
            _cache_cur_concat,
            name_hint="te_cache_cur_concat_value",
            args=[v_cached, v_cur, c],
        )

        self.k_cache.override(
            op.squeeze(k_cur, axis=0), self.sliding_window_size, self.attention_sink_size
        )
        self.v_cache.override(
            op.squeeze(v_cur, axis=0), self.sliding_window_size, self.attention_sink_size
        )

        return k, v

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rolling_cache_len: tir.Var,  # Number of elements currently in the cache.
        kv_seq_len: tir.Var,  # Equals to ``seq_len + rolling_cache_len``.
        cache_offset: tir.Var,
    ):
        """Forward pass of MistralAttention, performing QKV."""
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."
        qkv_cur = self.qkv_proj(hidden_states)
        qkv_cur = op.reshape(qkv_cur, (b, s, h_q + 2 * h_kv, d))
        q, k_cur, v_cur = op.split(qkv_cur, [h_q, h_q + h_kv], axis=2)
        k, v = self.interleave_kv(k_cur, v_cur, kv_seq_len, rolling_cache_len, cache_offset)
        q, k = self.rotary_embedding(q, k, rolling_cache_len)
        output = op_ext.attention(q, k, v, attention_mask)
        return self.o_proj(output)


class RollingKVCacheWithSinks(nn.KVCache):
    """
    Rolling buffer cache implementation.
    """

    cache: Optional[rx.Var]

    def override(self, new_element: Tensor, max_cache_size: int, attention_sink_size: int) -> None:
        """
        Override cache elements in RollingKVCacheWithSinks.

        Parameters
        ----------
        new_element : Tensor
            The new tensor to append.

        max_cache_size : int
            Max size of the cache.

        attention_sink_size : int
            Number of stored attention sinks.
        """
        if new_element.dtype != self.dtype:
            raise TypeError(
                f'RollingKVCacheWithSinks has been set to use dtype "{self.dtype}", '
                f'but got "{new_element.dtype}"'
            )
        self.cache = rx.BlockBuilder.current().emit(
            rx.Call(
                rx.extern("vm.builtin.attention_kv_cache_window_override_with_sinks"),
                args=[
                    self.cache,
                    new_element._expr,  # pylint: disable=protected-access
                    rx.PrimValue(max_cache_size),
                    rx.PrimValue(attention_sink_size),
                ],
                sinfo_args=[rx.ObjectStructInfo()],
            )
        )


class MistralDecoderLayer(nn.Module):
    """Exact same as LlamaDecoderLayer."""

    def __init__(self, config: MistralConfig, rotary_embedding: RotaryEmbedding):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = MistralAttention(config, rotary_embedding)
        self.mlp = MistralMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.mlp.gate_up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.residual_scale = 40 if config.hidden_size == 2304 else 52
        _set_tp()

    def forward(  # pylint: disable=too-many-arguments
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """Forward pass of a decoder layer; calculate attention, and add an residual connection."""

        def _apply_residual(out, residual):
            scale_out = out * 1.4 / self.residual_scale**0.5
            if self.tensor_parallel_shards > 1:
                return op.ccl_allreduce(scale_out + residual / self.tensor_parallel_shards, "sum")
            return scale_out + residual

        out = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            rolling_cache_len,
            kv_seq_len,
            cache_offset,
        )
        hidden_states = _apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = _apply_residual(out, residual=hidden_states)
        return hidden_states


class MistralModel(nn.Module):
    """Exact same as LlamaModel."""

    def __init__(self, config: MistralConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        rotary_embedding = RotaryEmbedding(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, rotary_embedding) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(  # pylint: disable=too-many-arguments
        self,
        hidden_states: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
        attention_mask: Tensor,
    ):
        """Forward pass of the model, passing through all decoder layers."""
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, rolling_cache_len, kv_seq_len, cache_offset
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

class MistralForCasualLM(nn.Module):
    """Same as LlamaForCausalLM, except for the use of sliding window attention."""

    def __init__(self, config: MistralConfig):
        self.model = MistralModel(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.sliding_window_size = config.sliding_window_size
        self.scale_embed = 12.0
        self.scale_lmhead = 9.0 if config.hidden_size == 2304 else 6.0
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(
        self,
        inputs: Tensor,
    ):
        if self.model.tensor_parallel_shards > 1:
            inputs = op.ccl_broadcast_from_worker0(inputs)
        inputs = self.model.embed_tokens(inputs) * self.scale_embed
        return inputs

    def forward(  # pylint: disable=too-many-arguments
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
        attention_mask: Tensor,
    ):
        """Forward pass."""

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(
            inputs, rolling_cache_len, kv_seq_len, cache_offset, attention_mask
        )
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        w = op.permute_dims(self.model.embed_tokens.weight)
        logits = op.matmul(hidden_states / self.scale_lmhead, w)
        # logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """
        Prefilling the prompt.

        Parameters
        ----------
        inputs: Tensor
            Input tokens, having ``seq_len`` number of tokens.

        rolling_cache_len: tir.Var
            Number of elements currently in the cache.

        kv_seq_len: tir.Var
            Equals to ``seq_len + rolling_cache_len``.

        cache_offset: tir.Var
            Next position to be overrided on the rolling kv cache.
        """
        def _sliding_window_attention_mask(
            batch_size, seq_len, rolling_cache_len, kv_seq_len, sliding_window_size
        ):
            # See `tests/legacy-python/test_sliding_window_mask.py` for its behavior
            return te.compute(
                (batch_size, 1, seq_len, kv_seq_len),
                lambda b, _, i, j: tir.Select(
                    tir.all(
                        i + rolling_cache_len >= j, i + rolling_cache_len - j < sliding_window_size
                    ),
                    tir.max_value(self.dtype),
                    tir.min_value(self.dtype),
                ),
                name="sliding_window_attention_mask_prefill",
            )

        batch_size, seq_len = inputs.shape
        attention_mask = op.tensor_expr_op(
            _sliding_window_attention_mask,
            name_hint="sliding_window_attention_mask_prefill",
            args=[
                batch_size,
                seq_len,
                rolling_cache_len,
                kv_seq_len,
                self.sliding_window_size,
            ],
        )
        inputs = self.embed(inputs)
        return self.forward(inputs, rolling_cache_len, kv_seq_len, cache_offset, attention_mask)

    def prefill_embed(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """
        Prefilling the prompt.

        Parameters
        ----------
        inputs: Tensor
            Input tokens, having ``seq_len`` number of embeddings.

        rolling_cache_len: tir.Var
            Number of elements currently in the cache.

        kv_seq_len: tir.Var
            Equals to ``seq_len + rolling_cache_len``.

        cache_offset: tir.Var
            Next position to be overrided on the rolling kv cache.
        """
        def _sliding_window_attention_mask(
            batch_size, seq_len, rolling_cache_len, kv_seq_len, sliding_window_size
        ):
            # See `tests/legacy-python/test_sliding_window_mask.py` for its behavior
            return te.compute(
                (batch_size, 1, seq_len, kv_seq_len),
                lambda b, _, i, j: tir.Select(
                    tir.all(
                        i + rolling_cache_len >= j, i + rolling_cache_len - j < sliding_window_size
                    ),
                    tir.max_value(self.dtype),
                    tir.min_value(self.dtype),
                ),
                name="sliding_window_attention_mask_prefill",
            )

        batch_size, seq_len, _ = inputs.shape
        attention_mask = op.tensor_expr_op(
            _sliding_window_attention_mask,
            name_hint="sliding_window_attention_mask_prefill",
            args=[
                batch_size,
                seq_len,
                rolling_cache_len,
                kv_seq_len,
                self.sliding_window_size,
            ],
        )
        return self.forward(inputs, rolling_cache_len, kv_seq_len, cache_offset, attention_mask)

    def decode(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """Decoding step."""
        batch_size, seq_len = inputs.shape
        attention_mask = op.full(
            shape=[batch_size, 1, seq_len, kv_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        inputs = self.embed(inputs)
        return self.forward(inputs, rolling_cache_len, kv_seq_len, cache_offset, attention_mask)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        """Softmax."""
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class ResamplerAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config):
        self.hidden_size = config.output_dim
        self.head_dim = 128
        self.num_heads = self.hidden_size // self.head_dim // config.tensor_parallel_shards
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor,
    ):
        """Forward pass of MistralAttention, performing QKV."""
        d, h = self.head_dim, self.num_heads
        b, sq, _ = q.shape
        _, sk, _ = k.shape
        assert b == 1, "Only support batch size 1 at this moment."
        q = self.q_proj(q).reshape(b, sq, h, d)
        k = self.k_proj(k).reshape(b, sk, h, d)
        v = self.v_proj(v).reshape(b, sk, h, d)
        output = op_ext.attention(q, k, v, attention_mask)
        return self.out_proj(output)

class Resampler(nn.Module):
    def __init__(self, config: ViTConfig):
        self.num_queries = config.num_query
        self.embed_dim = config.output_dim
        self.kv_dim = config.hidden_size
        self.image_len = config.image_len

        self.pos_embed = nn.Parameter((self.num_queries, self.embed_dim))
        #self.pos_embed_k = nn.Parameter((self.image_len, self.embed_dim))
        self.pos_embed_k = nn.Parameter((448*448//14//14, self.embed_dim))
        self.query = nn.Parameter((self.num_queries, self.embed_dim))
        self.kv_proj = nn.Linear(self.kv_dim, self.embed_dim, bias=False)
        self.ln_q = nn.LayerNorm(self.embed_dim, config.norm_eps)
        self.ln_kv = nn.LayerNorm(self.embed_dim, config.norm_eps)
        self.attn = ResamplerAttention(config)
        self.ln_post = nn.LayerNorm(self.embed_dim, config.norm_eps)
        self.proj = nn.Parameter((self.embed_dim, self.embed_dim))
        self.dtype = 'float32'


    def forward(
        self, 
        x : Tensor,
        tgt_size,
    ):
        # print(self.pos_embed_k)
        # print(self.pos_embed_k.data)
        # print(tgt_size)
        #if self.pos_embed_k.data is None:
        #    pos_embed = op.zeros((tgt_size[0] * tgt_size[1], self.embed_dim), dtype=self.pos_embed_k.dtype)
        #else:
        #    #pos_embed = self.pos_embed_k.data().reshape(32, 32, self.embed_dim)[:tgt_size[0], :tgt_size[1]].reshape(tgt_size[0] * tgt_size[1], self.embed_dim)
        #    op.take(self.pos_embed_k)
        pos_embed = self.pos_embed_k.reshape(32, 32, self.embed_dim)
        indices_x = nn.core.wrap_nested(tvm.relax.op.arange(0, tgt_size[0], dtype="int32"), "arange")
        indices_y = nn.core.wrap_nested(tvm.relax.op.arange(0, tgt_size[1], dtype="int32"), "arange")
        print(pos_embed)
        print(indices_x)
        pos_embed = op.take(pos_embed, indices_x, axis=0)
        pos_embed = op.take(pos_embed, indices_y, axis=1).reshape(tgt_size[0] * tgt_size[1], self.embed_dim)

        x = self.kv_proj(x)
        x = self.ln_kv(x)

        q = self.ln_q(self.query)

        def _attention_mask(
            batch_size, q_len, k_len,
        ):
            # See `tests/legacy-python/test_sliding_window_mask.py` for its behavior
            return te.compute(
                (batch_size, 1, q_len, k_len),
                lambda b, _, i, j: tir.max_value(self.dtype),
                name="_attention_mask",
            )

        attention_mask = op.tensor_expr_op(
            _attention_mask,
            name_hint="_attention_mask",
            args=[
                1,
                self.num_queries,
                self.image_len,
            ],
        )

        # print(x.shape, pos_embed.shape, q.shape, self.pos_embed.shape)
        x = self.attn(
            (q + self.pos_embed).reshape(1, q.shape[0], q.shape[1]),
            x + pos_embed.reshape(1, pos_embed.shape[0], pos_embed.shape[1]),
            x,
            attention_mask
        )
        # print("end")

        x = self.ln_post(x)

        x = op.matmul(x, self.proj)
        return x

class VisMiniCPM(nn.Module):
    def __init__(self, config: MistralConfig):
        vit_config = ViTConfig()
        self.llm = MistralForCasualLM(config)
        self.vpm = ViT(vit_config)
        self.resampler = Resampler(vit_config)
        self.dtype = "float32"

    def image(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        inputs = (inputs.astype(self.dtype) / 255. - 0.5) / 0.5
        shape = inputs.shape
        print("before resampler: ", shape)
        inputs = self.vpm(inputs)
        inputs = self.resampler(inputs, ((shape[-2] + 13) // 14, (shape[-1] + 13) // 14))
        return self.llm.prefill_embed(inputs, rolling_cache_len, kv_seq_len, cache_offset)

    def prefill(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        return self.llm.prefill(inputs, rolling_cache_len, kv_seq_len, cache_offset)

    def decode(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        return self.llm.decode(inputs, rolling_cache_len, kv_seq_len, cache_offset)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        """Softmax."""
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1
        image_size = 448
        mod_spec = {
            "image": {
                "inputs": nn.spec.Tensor([batch_size, 3, image_size, image_size], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


