import jax
from array_types import Array

from typing import Callable, Iterable

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

    def __setattr__(self, name: str, value: "ParamTree") -> None:
        return setattr(self.params.name, name, value)

class ParamTree(dict):
    def __init__(self):
        super().__init__()
        self.training = True

    @property
    def children(self):
        return ChildrenParams(self)

    def __getattr__(self, key: str) -> Array:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ParamTree' object has no attribute '{key}'")

    def __setattr__(self, name: str, value: Array) -> None:
        return super().__setattr__(name, value)

    def param(self, name: str, key: jax.Array, init: Callable[[jax.Array], Array]) -> Array:
        if name not in self:
            self[name] = init(key)
        return self[name]