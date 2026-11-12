import jax.numpy as jnp

from funcax.functional import attn
from funcax.nn.modules import make_linear, make_mlp, make_activation, make_layer_norm, ModuleFn
from funcax.functional.norm import layer_norm

from funcax.types import Array, ParamTree

def make_attn():
    def apply(params: ParamTree, q: Array, k: Array, v: Array):
        return attn(params, q, k, v)

    def init(key: Array):
        params = ParamTree()
        params.children.softmax = ParamTree()
        return key, params

    return apply, init

def split_heads(array: Array, attn_heads: int, head_dim: int):
    return array.reshape(*array.shape[:-2], array.shape[-2], attn_heads, head_dim).swapaxes(-3, -2)

def merge_heads(array: Array, hidden_dim: int):
    array = array.swapaxes(-3, -2)
    return array.reshape(*array.shape[:-2], hidden_dim)

def make_self_attention(
        activation_factory: ModuleFn,
        in_dim: int,
        attn_heads: int,
        hidden_dim: int,
        hidden_scale: int
    ):
    linear_apply, linear_init = make_linear()
    mlp_apply, mlp_init = make_mlp(activation_factory)
    norm_apply, norm_init = make_layer_norm()
    attn_apply, attn_init = make_attn()

    hidden_dim = in_dim * attn_heads
    head_dim = in_dim

    def apply(params: ParamTree, array: Array):
        out = linear_apply(params.children.linear, array)
        out = norm_apply(params.children.norm, out)

        q, k, v = jnp.split(out, 3, axis=-1)
        q, k, v = [split_heads(x, attn_heads, head_dim) for x in (q, k, v)]

        out = attn_apply(params.children.attn, q, k, v)
        out = merge_heads(out, hidden_dim)
        return mlp_apply(params.children.mlp, out)

    def init(key: Array):
        params = ParamTree()
        key, params.children.linear = linear_init(key, in_dim, hidden_dim * 3)
        key, params.children.mlp = mlp_init(key, hidden_dim, hidden_dim * hidden_scale, in_dim)
        
        key, params.children.norm = norm_init(key, hidden_dim * 3)
        key, params.children.attn = attn_init(key)

        return key, params
        
    return apply, init