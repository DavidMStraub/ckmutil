"""JAX-compatible versions of all ckmutil functions.

Requires JAX (``pip install jax`` or ``pip install ckmutil[jax]``).

Functions are not pre-jitted so that callers retain full control over
``jax.jit``, ``jax.vmap``, ``jax.grad``, ``static_argnums``, etc.
"""

try:
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError(
        "JAX is required to use ckmutil.jax. "
        "Install it with: pip install jax"
    ) from e

from functools import partial

from ckmutil._ckm_impl import (
    _ckm_standard,
    _ckm_wolfenstein,
    _ckm_tree,
    _ckm_beta_gamma,
    _gamma_to_delta,
    _beta_gamma_to_delta,
    _tree_to_standard,
    _standard_to_tree,
    _beta_gamma_to_standard,
    _standard_to_beta_gamma,
    _wolfenstein_to_standard,
    _standard_to_wolfenstein,
    _tree_to_wolfenstein,
    _wolfenstein_to_tree,
)
from ckmutil._diag_impl import _msvd

ckm_standard         = partial(_ckm_standard, jnp)
ckm_wolfenstein      = partial(_ckm_wolfenstein, jnp)
ckm_tree             = partial(_ckm_tree, jnp)
ckm_beta_gamma       = partial(_ckm_beta_gamma, jnp)

gamma_to_delta       = partial(_gamma_to_delta, jnp)
beta_gamma_to_delta  = partial(_beta_gamma_to_delta, jnp)

tree_to_standard     = partial(_tree_to_standard, jnp)
standard_to_tree     = partial(_standard_to_tree, jnp)
beta_gamma_to_standard  = partial(_beta_gamma_to_standard, jnp)
standard_to_beta_gamma  = partial(_standard_to_beta_gamma, jnp)
wolfenstein_to_standard = partial(_wolfenstein_to_standard, jnp)
standard_to_wolfenstein = partial(_standard_to_wolfenstein, jnp)
tree_to_wolfenstein  = partial(_tree_to_wolfenstein, jnp)
wolfenstein_to_tree  = partial(_wolfenstein_to_tree, jnp)

msvd                 = partial(_msvd, jnp)
