import jax
import jax.numpy as jnp

from typing import Callable, Any

class ChildrenParams:
    def __init__(self, params: "ParamTree"):
        self.params = params

    def __getattr__(self, name: str) -> "ParamTree":
        try:
            child = self.params[name]
        except KeyError:
            raise AttributeError(f"'ParamTree' has no child named {name}")

        if not isinstance(child, "ParamTree"):
            raise AttributeError(f"{name} is a parameter not a child ParamTree")

        return child

class ParamTree(dict):
    @property
    def children(self):
        return ChildrenParams(self)

    def __getattr__(self, key: Any) -> Array:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ParamTree' object has no attribute '{key}'")

    def param(self, name: str, key: jax.Array, init: Callable[[jax.Array], Array]) -> Array:
        if name not in self:
            self[name] = init(key)
        return self[name]