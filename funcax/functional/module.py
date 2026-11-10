import jax
import jax.numpy as jnp

from .activation import sigmoid, tanh
from funcax.types import Array, ParamTree

def identity(_: ParamTree, array: Array):
    return array

def linear(params: ParamTree, array: Array):
    return array @ params.W + params.b

def lstm(params: ParamTree, array: Array, hidden: Array, state: Array):
    stacked = jnp.concatenate([array, hidden], axis=0)

    forget_gate = sigmoid(params.children.forget_sigmoid, params.W_f @ stacked + params.b_f)
    input_gate = sigmoid(params.children.input_gate, params.W_i @ stacked + params.b_i)

    candidate_cell_state = tanh(params.children.state_tanh, params.W_c @ stacked + params.b_c)
    state = forget_gate * state + input_gate * candidate_cell_state

    output_gate = sigmoid(params.children.output_sigmoid, params.W_o @ stacked + params.b_o)
    hidden = output_gate * tanh(params, state)
    return hidden, state

def convolution(params: ParamTree, array: Array):
    n = params.W.ndim - 2
    out = jax.lax.conv_general_dilated(
        jnp.expand_dims(array, axis=0),
        params.W,
        (1,) * n,
        "SAME"
    )
    return out.squeeze(axis=0) + params.b.reshape((-1,) + (1,) * n)

def embedding(params: ParamTree, array: Array):
    return jnp.take(params.E, array, axis=0)

def dropout(params: ParamTree, array: Array, drop: float = 0.1):
    if not params.training or drop == 0.0:
        return array

    keep = 1.0 - drop
    mask = jax.random.bernoulli(params.key, keep, array.shape)
    return jnp.where(mask, array / keep, 0.0)