import jax.numpy as jnp

from funcax.types import Array, ParamTree
from funcax.functional.activation import softmax

def attn(params: ParamTree, q: Array, k: Array, v: Array):
    scores = q @ k.swapaxes(-1, -2) / jnp.sqrt(q.shape[-1])
    weights = softmax(params.children.softmax, scores)
    return weights @ v