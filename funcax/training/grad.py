import jax

from funcax.types import ParamTree, Array

from typing import Callable, Any

def apply_grads(params: ParamTree, grads: Any, lr: float) -> ParamTree:
    updated = ParamTree()

    for name, param in params.items():
        grad = grads[name]
        if isinstance(param, ParamTree):
            updated[name] = apply_grads(param, grad, lr)
        else:
            updated[name] = param - lr * grad

    return updated

def train_fn(
        forward: Callable[[ParamTree, Array], Array],
        loss_callback: Callable[[Array, Array], Array],
        apply_grads: Callable[[ParamTree, Any, float], ParamTree],
        lr: float
    ):
    def loss_fn(params: ParamTree, features: Array, target: Array):
        pred = forward(params, features)
        return loss_callback(pred, target)

    @jax.jit
    def apply_fn(params: ParamTree, features: Array, target: Array):
        loss, grads = jax.value_and_grad(loss_fn)(params, features, target)
        params = apply_grads(params, grads, lr)
        return params, loss

    return apply_fn