from funcax.functional import embedding, sinusoidal_embedding, rotary_embedding
from funcax.types import Array, ParamTree, make_param

def make_embedding(num_embeddings: int, embedding_dim: int):
    def apply(params: ParamTree, array: Array):
        return embedding(params, array)

    def init(key: Array):
        params = ParamTree()
        key, params.E = make_param(key, (num_embeddings, embedding_dim), std=0.02)
        return key, params

    return apply, init

def make_sinusoidal_embedding():
    def apply(params: ParamTree, array: Array):
        return sinusoidal_embedding(params, array)

    def init(key: Array):
        return key, ParamTree()

    return apply, init

def make_rotary_embedding():
    def apply(params: ParamTree, array: Array):
        return rotary_embedding(params, array)

    def init(key: Array):
        return key, ParamTree()

    return apply, init