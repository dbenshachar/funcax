import jax.numpy as jnp

from funcax.types import Array, ParamTree

def layer_norm(params: ParamTree, array: Array, eps: float = 1e-5):
    mean = array.mean(axis=-1, keepdims=True)
    var = array.var(axis=-1, keepdims=True)
    return params.gamma * (array - mean) / jnp.sqrt(var + eps) + params.beta