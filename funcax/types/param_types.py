import jax
from funcax.types.array_types import Array

from typing import Callable, Sequence

class ChildrenParams:
    def __init__(self, params: "ParamTree"):
        object.__setattr__(self, "params", params)

    def __setattr__(self, name: str, value: "ParamTree") -> None:
        self.params[name] = value

    def __getattr__(self, name: str) -> "ParamTree":
        try:
            child = self.params[name]
        except KeyError:
            raise AttributeError(f"'ParamTree' has no child named {name}")

        if not isinstance(child, ParamTree):
            raise AttributeError(f"{name} is a parameter not a child ParamTree")

        return child

@jax.tree_util.register_pytree_node_class
class ParamTree(dict):
    def __init__(self):
        super().__init__()
        object.__setattr__(self, "training", True)

    @property
    def children(self):
        return ChildrenParams(self)

    def __getattr__(self, key: str) -> Array:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ParamTree' object has no attribute '{key}'")

    def __setattr__(self, name: str, value: Array) -> None:
        if name == "training":
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def tree_flatten(self):
        keys = tuple(self.keys())
        values = tuple(self[key] for key in keys)
        return values, (keys, self.training)

    @classmethod
    def tree_unflatten(cls, aux_data, values):
        keys, training = aux_data
        params = cls()
        object.__setattr__(params, "training", training)
        params.update(zip(keys, values))
        return params

    def param(self, name: str, key: jax.Array, init: Callable[[jax.Array], Array]) -> Array:
        if name not in self:
            self[name] = init(key)
        return self[name]

def make_param(key: Array, shape: Sequence[int], mean: float | None = None, std: float | None = None):
    scale = 1.0 if std is None else std
    location = 0.0 if mean is None else mean
    key, k = jax.random.split(key)
    param = jax.random.normal(k, shape) * scale + location
    return key, param