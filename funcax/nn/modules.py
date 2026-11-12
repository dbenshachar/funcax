import jax
import jax.numpy as jnp

from funcax.types import Array, ParamTree, make_param
from funcax.functional import identity, linear, dropout, convolution, lstm

from typing import Callable, Sequence

type ModuleFn = Callable[
    [],
    tuple[
        Callable[[ParamTree, Array], Array],
        Callable[[Array], tuple[Array, ParamTree]],
    ],
]

def make_identity():
    def apply(params: ParamTree, array: Array):
        return identity(params, array)

    def init(_: Array):
        return _, ParamTree()
        
    return apply, init

def make_activation(activation: Callable[[ParamTree, Array], Array]):
    def apply(params: ParamTree, array: Array):
        return activation(params, array)

    def init(_: Array):
        return _, ParamTree()
    
    return apply, init

def make_linear():
    def apply(params: ParamTree, array: Array):
        return linear(params, array)

    def init(key: Array, in_dim: int, out_dim: int):
        params = ParamTree()
        key, params.W = make_param(key, (in_dim, out_dim))
        key, params.b = make_param(key, (out_dim,))
        return key, params

    return apply, init

def make_mlp(activation_factory: ModuleFn):
    dense1_apply, dense1_init = make_linear()
    dense2_apply, dense2_init = make_linear()
    activation_apply, activation_init = activation_factory()

    def apply(params: ParamTree, array: Array):
        out = dense1_apply(params.children.dense1, array)
        out = activation_apply(params.children.activation, out)
        out = dense2_apply(params.children.dense2, out)
        return out

    def init(key: Array, in_dim: int, hidden_dim: int, out_dim: int):
        params = ParamTree()
        key, params.children.dense1 = dense1_init(key, in_dim, hidden_dim)
        key, params.children.activation = activation_init(key)
        key, params.children.dense2 = dense2_init(key, hidden_dim, out_dim)
        return key, params

    return apply, init

def make_dropout(drop: float = 0.1):
    def apply(params: ParamTree, array: Array):
        return dropout(params, array, drop=drop)
    
    def init(_: Array):
        return _, ParamTree()

    return apply, init

def make_convolution():
    def apply(params: ParamTree, array: Array):
        return convolution(params, array)

    def init(key: Array, in_channels: int, out_channels: int, kernel_shape: Sequence[int]):
        params = ParamTree()
        key, params.W = make_param(key, (out_channels, in_channels, *kernel_shape))
        key, params.b = make_param(key, (out_channels,))
        return key, params

    return apply, init

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

    def init(key: Array, in_dim: int, hidden_dim: int):
        params = ParamTree()

        for layer in range(layers):
            layer_params = ParamTree()
            layer_in_dim = in_dim if layer == 0 else hidden_dim
            stacked_dim = layer_in_dim + hidden_dim

            key, layer_params.W_f = make_param(key, (hidden_dim, stacked_dim))
            key, layer_params.b_f = make_param(key, (hidden_dim,))

            key, layer_params.W_i = make_param(key, (hidden_dim, stacked_dim))
            key, layer_params.b_i = make_param(key, (hidden_dim,))

            key, layer_params.W_c = make_param(key, (hidden_dim, stacked_dim))
            key, layer_params.b_c = make_param(key, (hidden_dim,))

            key, layer_params.W_o = make_param(key, (hidden_dim, stacked_dim))
            key, layer_params.b_o = make_param(key, (hidden_dim,))

            layer_params["forget_sigmoid"] = ParamTree()
            layer_params["input_gate"] = ParamTree()
            layer_params["state_tanh"] = ParamTree()
            layer_params["output_sigmoid"] = ParamTree()

            params[f"lstm_{layer}"] = layer_params

        return key, params

    return apply, init