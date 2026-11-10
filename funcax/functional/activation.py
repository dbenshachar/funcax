import jax

from funcax.types import Array, ParamTree
        
def relu(_: ParamTree, array: Array):
    return jax.nn.relu(array)

def gelu(_: ParamTree, array: Array):
    return jax.nn.gelu(array)

def silu(_: ParamTree, array: Array):
    return jax.nn.gelu(array)

def tanh(_: ParamTree, array: Array):
    return jax.nn.tanh(array)

def sigmoid(_: ParamTree, array: Array):
    return jax.nn.sigmoid(array)

def softmax(_: ParamTree, array: Array):
    return jax.nn.softmax(array)