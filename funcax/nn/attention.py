import jax.numpy as jnp

from funcax.functional import attn
from funcax.nn.modules import make_linear, make_mlp, make_activation, ModuleFn
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

def make_self_attention(activation_factory: ModuleFn):
    linear_apply, linear_init = make_linear()
    mlp_apply, mlp_init = make_mlp(activation_factory)
    norm_apply, norm_init = make_activation(layer_norm)
    attn_apply, attn_init = make_attn()

    def apply(params: ParamTree, array: Array):
        out = linear_apply(params.children.linear, array)
        out = norm_apply(params.children.norm, out)
        q, k, v = jnp.split(out, 3, axis=-1)
        out = attn_apply(params.children.attn, q, k, v)
        out = mlp_apply(params.children.mlp, out)
        return out

    def init(key: Array, in_dim: int, attn_heads: int, hidden_scale: int):
        params = ParamTree()
        hidden_dim = in_dim * attn_heads
        key, params.children.linear = linear_init(key, in_dim, hidden_dim * 3)
        key, params.children.mlp = mlp_init(key, hidden_dim, hidden_dim * hidden_scale, hidden_dim)
        
        key, params.children.norm = norm_init(key)
        key, params.children.attn = attn_init(key)

        return key, params
        
    return apply, init

def make_cross_attention(activation_factory: ModuleFn):
    q_linear_apply, q_linear_init = make_linear()
    kv_linear_apply, kv_linear_init = make_linear()
    mlp_apply, mlp_init = make_mlp(activation_factory)
    norm_apply, norm_init = make_activation(layer_norm)
    attn_apply, attn_init = make_attn()

    def apply(params: ParamTree, q: Array, kv: Array):
        q = q_linear_apply(params.children.q_linear, q)
        kv = kv_linear_apply(params.children.kv_linear, kv)
        k, v = jnp.split(kv, 2, axis=-1)
        q, k, v = norm_apply(params.children.norm, q), norm_apply(params.children.norm, k), norm_apply(params.children.norm, v)
        out = attn_apply(params.children.attn, q, k, v)
        out = mlp_apply(params.children.mlp, out)
        return out

    def init(key: Array, in_dim: int, attn_heads: int, hidden_scale: int):
        params = ParamTree()
        hidden_dim = in_dim * attn_heads
        key, params.children.q_linear = q_linear_init(key, in_dim, hidden_dim)
        key, params.children.kv_linear = kv_linear_init(key, in_dim, hidden_dim * 2)
        key, params.children.mlp = mlp_init(key, hidden_dim, hidden_dim * hidden_scale, hidden_dim)
        
        key, params.children.norm = norm_init(key)
        key, params.children.attn = attn_init(key)

        return key, params
        
    return apply, init