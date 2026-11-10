import jax.numpy as jnp

from funcax.types import Array

def mse_loss(pred: Array, target: Array):
    return (pred - target) ** 2

def binary_cross_entropy_loss(pred: Array, target: Array):
    return -(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))

def cross_entropy_loss(pred: Array, target: Array):
    return -jnp.log(pred[target])