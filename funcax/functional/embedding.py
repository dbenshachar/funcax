import jax.numpy as jnp

from funcax.types import Array, ParamTree

def embedding(params: ParamTree, array: Array):
    return jnp.take(params.E, array, axis=0)

def sinusoidal_embedding(_: ParamTree, array: Array):
    seq_len, dim = array.shape[-2], array.shape[-1]
    pos = jnp.arange(seq_len)[:, None]
    inv_freq = 10000.0 ** (-jnp.arange(0, dim, 2) / dim)
    angles = pos * inv_freq[None, :]
    pe = jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(seq_len, dim)
    return array + pe

def rotary_embedding(_: ParamTree, array: Array):
    seq_len, dim = array.shape[-2], array.shape[-1]
    pos = jnp.arange(seq_len)[:, None]
    inv_freq = 10000.0 ** (-jnp.arange(0, dim, 2) / dim)
    angles = pos * inv_freq[None, :]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    x1, x2 = jnp.split(array, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)