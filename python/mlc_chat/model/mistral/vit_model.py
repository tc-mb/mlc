"""
Implementation for ViT architecture.
"""
import dataclasses
from typing import Any, Dict, Optional

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

@dataclasses.dataclass
class ViTConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the ViT model."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 26
    norm_eps: float = 1e-6
    num_key_value_heads: int = 16
    head_dim: int = 72
    tensor_parallel_shards: int = 1
    channels_in : int = 3
    image_size : int = 448
    conv_size : int = 14
    image_len : int = 256 # (224/14)**2
    num_query : int = 64
    output_dim : int = 2304
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size

class ViTMLP(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.fc1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.intermediate_size,
            bias=True,
        )
        self.fc2 = nn.Linear(self.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: Tensor):
        return self.fc2(op.gelu(self.fc1(x)))

class ViTAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: ViTConfig):
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.qkv = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=True,
        )
        self.proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ):
        """Forward pass of MistralAttention, performing QKV."""
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."
        qkv_cur = self.qkv(hidden_states)
        qkv_cur = op.reshape(qkv_cur, (b, s, h_q + 2 * h_kv, d))
        q, k, v = op.split(qkv_cur, [h_q, h_q + h_kv], axis=2)
        output = op_ext.attention(q, k, v, attention_mask)
        return self.proj(output)

class ViTDecoderLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        self.norm_eps = config.norm_eps
        self.attn = ViTAttention(config)
        self.mlp = ViTMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, self.norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, self.norm_eps)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.attn.num_q_heads * hd
            k = self.attn.num_kv_heads * hd
            v = self.attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(self.attn.qkv, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.attn.proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.mlp.fc1, tp.ShardSingleDim("_shard_mlp_up", dim=0))
            _set(self.mlp.fc2, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(  # pylint: disable=too-many-arguments
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ):
        """Forward pass of a decoder layer; calculate attention, and add an residual connection."""

        def _apply_residual(out, residual):
            if self.tensor_parallel_shards > 1:
                return op.ccl_allreduce(out + residual / self.tensor_parallel_shards, "sum")
            return out + residual

        out = self.attn(self.norm1(hidden_states), attention_mask)
        hidden_states = _apply_residual(out, residual=hidden_states)
        out = self.mlp(self.norm2(hidden_states))
        hidden_states = _apply_residual(out, residual=hidden_states)
        return hidden_states

class Conv2d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride):
        self.weight = nn.Parameter((chan_out, chan_in, kernel_size, stride))
        self.bias = nn.Parameter((chan_out,))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(
        self,
        x: Tensor,
    ):
        x = op.conv2d(x, self.weight, stride=self.stride)
        x = x + self.bias.reshape(1, self.bias.shape[0], 1, 1)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size):
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.proj = Conv2d(chan_in, chan_out, kernel_size, kernel_size)

    def forward(
        self,
        inputs: Tensor
    ):
        embed = self.proj(inputs)
        embed = embed.reshape(embed.shape[0], embed.shape[1], embed.shape[2]*embed.shape[3])
        return embed.permute_dims(0, 2, 1)

class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.patch_embed = PatchEmbed(config.channels_in, config.hidden_size, config.conv_size)
        self.pos_embed = nn.Parameter((1, config.image_len, config.hidden_size))
        self.blocks = nn.ModuleList(
            [ViTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_eps)
        self.config = config
        self.dtype = "float32"

    def forward(
        self,
        inputs: Tensor,
    ):
        def _vit_attention_mask(
            batch_size, seq_len,
        ):
            # See `tests/legacy-python/test_sliding_window_mask.py` for its behavior
            return te.compute(
                (batch_size, 1, seq_len, seq_len),
                lambda b, _, i, j: tir.max_value(self.dtype),
                name="vit_attention_mask",
            )

        batch_size, _, _, _ = inputs.shape
        attention_mask = op.tensor_expr_op(
            _vit_attention_mask,
            name_hint="vit_attention_mask",
            args=[
                batch_size,
                self.config.image_len,
            ],
        )

        hidden_states = self.patch_embed(inputs)
        hidden_states = hidden_states + self.pos_embed
        for layer in self.blocks:
            hidden_states = layer(
                hidden_states, attention_mask
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states