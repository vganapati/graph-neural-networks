"""
Reference: https://jax.readthedocs.io/en/latest/quickstart.html
"""

import time
import jax.numpy as jnp
from jax import random
from jax import jit

key = random.key(1701)

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda*jnp.where(x>0, x, alpha*jnp.exp(x)-alpha)

selu_jit = jit(selu)

x = jnp.arange(5.0)
print(selu(x))

x = random.normal(key, (1000000,))

start = time.time()

selu(x).block_until_ready() # without block_until_ready, JAX can go ahead with the rest of the program with just the shape
end = time.time()
print("Time elapsed without jit is " + str(end-start) + " seconds.")

start = time.time()
_ = selu_jit(x).block_until_ready() # compiles on first call
end = time.time()
print("Time elapsed to compile is " + str(end-start) + " seconds.")

start = time.time()
selu_jit(x). block_until_ready()
end = time.time()
print("Time elapsed with compiled jit is " + str(end-start) + " seconds.")


# Derivatives with jax.grad()

from jax import grad

def sum_logistic(x):
    return jnp.sum(1.0/(1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = jit(grad(sum_logistic))
print(derivative_fn(x_small))
breakpoint()