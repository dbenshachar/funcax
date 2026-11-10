import jax
import jax.numpy as jnp

from funcax.types import Array, ParamTree
from funcax.functional import identity, linear, embedding, dropout, convolution, lstm

from typing import Callable

def make_identiy():
    def apply(params: ParamTree, array: Array):
        return identity(params, array)
    return apply

def make_activation(activation: Callable[[Array], Array]):
    def apply(_: ParamTree, array: Array):
        return activation(array)
    return apply

def make_linear():
    def apply(params: ParamTree, array: Array):
        return linear(params, array)
    return apply

def make_mlp(hidden: int, dim: int, make_activation: Callable[[], Callable[[ParamTree, Array], Array]]):
    dense1 = make_linear()
    activation = make_activation()
    dense2 = make_linear()
    def apply(params: ParamTree, array: Array):
        out = dense1(params.children.dense1, array)
        out = activation(params.children.activation, out)
        out = dense2(params.children.dense2, out)
        return out
    return apply

def make_dropout(drop: float = 0.1):
    def apply(params: ParamTree, array: Array):
        return dropout(params, array, drop=drop)
    return apply

def make_embedding():
    def apply(params: ParamTree, array: Array):
        return embedding(params, array)
    return apply

def make_convolution():
    def apply(params: ParamTree, array: Array):
        return convolution(params, array)
    return apply

def make_lstm(layers: int):
    def apply(params: ParamTree, array: Array):
        def step(memory: tuple[jnp.ndarray, jnp.ndarray], array: jnp.ndarray):
            hidden, state = memory
            new_hidden, new_state = [], []

            for layer in range(layers):
                h, c = lstm(getattr(params.children, f"lstm_{layer}"), array, hidden[layer], state[layer])
                new_hidden.append(h)
                new_state.append(c)
                array = h if layer == 0 else array + h

            hidden = jnp.stack(new_hidden)
            state = jnp.stack(new_state)
            return (hidden, state), array

        hidden = jnp.stack([jnp.zeros_like(params[f"{layer}.b_f"]) for layer in range(layers)])
        state = jnp.zeros_like(hidden)

        _, out = jax.lax.scan(step, (hidden, state), array)
        return out
    return apply