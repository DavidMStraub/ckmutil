# ckmutil

A package containing useful functions to deal with the Cabibbo-Kobayashi-Maskawa (CKM) quark mixing matrix or the Pontecorvo-Maki-Nakagawa-Sakata (PMNS) lepton mixing matrix in high energy physics.

[Documentation](https://flav-io.github.io/ckmutil/ckmutil/)

## JAX support

All functions in `ckmutil.ckm` and `msvd` from `ckmutil.diag` are available as JAX-compatible versions in `ckmutil.jax`:

```bash
pip install ckmutil[jax]
```

```python
import jax
import jax.numpy as jnp
from ckmutil.jax import ckm_standard, wolfenstein_to_standard

# differentiate
jax.grad(lambda d: jnp.abs(ckm_standard(t12, t13, t23, d)[0, 2]))(delta)

# jit-compile
ckm_fast = jax.jit(ckm_standard)

# vectorise over parameter arrays
jax.vmap(ckm_standard)(t12s, t13s, t23s, deltas)
```

Functions are not pre-jitted, so callers can apply `jax.jit`, `jax.vmap`, and `jax.grad` freely with their own options. `mixing_phases`, `rephase_standard`, and `rephase_pmns_standard` from `ckmutil.phases`, as well as `mtakfac` from `ckmutil.diag`, are not available in the JAX backend.
