import unittest
from math import radians, asin
import numpy as np
import numpy.testing as npt

try:
    import jax
    jax.config.update("jax_enable_x64", True)  # match numpy float64 precision
    import jax.numpy as jnp
    import ckmutil.jax as ckm_jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

import ckmutil.ckm as ckm_np
from ckmutil.diag import msvd as msvd_np

# Same parameter values as test_ckm.py
Vus = 0.22
Vub = 3.5e-3
Vcb = 4.0e-2
gamma = radians(70.)

s13 = Vub
c13 = np.sqrt(1 - s13**2)
s12 = Vus / c13
s23 = Vcb / c13
t12 = asin(s12)
t13 = asin(s13)
t23 = asin(s23)
delta = ckm_np.gamma_to_delta(t12, t13, t23, gamma)
laC = Vus
A = s23 / laC**2
Vcd_complex = -s12 * np.sqrt(1 - s23**2) - np.sqrt(1 - s12**2) * s23 * s13 * np.exp(1j * delta)
Vtd_complex = s12 * s23 - np.sqrt(1 - s12**2) * np.sqrt(1 - s23**2) * s13 * np.exp(1j * delta)
beta = np.angle(-Vcd_complex / Vtd_complex)
rho_minus_i_eta = s13 * np.exp(-1j * delta) / (A * laC**3)
rhobar_plus_i_etabar = (
    np.sqrt(1 - laC**2) * (rho_minus_i_eta.real - 1j * rho_minus_i_eta.imag)
    / (np.sqrt(1 - A**2 * laC**4) + np.sqrt(1 - laC**2) * A**2 * laC**4 * (rho_minus_i_eta.real - 1j * rho_minus_i_eta.imag))
)
rhobar = rhobar_plus_i_etabar.real
etabar = rhobar_plus_i_etabar.imag

Mc = np.array([[ 0.82469+0.62495j,  0.37768+0.28744j,  0.40011+0.93478j],
               [ 0.85475+0.60855j,  0.64045+0.93049j,  0.39019+0.6188j ],
               [ 0.68798+0.40478j,  0.38995+0.86032j,  0.20555+0.601j  ]])


@unittest.skipUnless(JAX_AVAILABLE, "JAX not installed")
class TestJaxMatchesNumpy(unittest.TestCase):
    """JAX results must be numerically identical to the numpy versions."""

    def test_ckm_standard(self):
        np_result = ckm_np.ckm_standard(t12, t13, t23, delta)
        jax_result = ckm_jax.ckm_standard(t12, t13, t23, delta)
        npt.assert_array_almost_equal(np.array(jax_result), np_result, decimal=12)

    def test_ckm_wolfenstein(self):
        np_result = ckm_np.ckm_wolfenstein(laC, A, rhobar, etabar)
        jax_result = ckm_jax.ckm_wolfenstein(laC, A, rhobar, etabar)
        npt.assert_array_almost_equal(np.array(jax_result), np_result, decimal=12)

    def test_ckm_tree(self):
        for order in (None, 0, 1, 2):
            with self.subTest(order=order):
                np_result = ckm_np.ckm_tree(Vus, Vub, Vcb, gamma, delta_expansion_order=order)
                jax_result = ckm_jax.ckm_tree(Vus, Vub, Vcb, gamma, delta_expansion_order=order)
                npt.assert_array_almost_equal(np.array(jax_result), np_result, decimal=12)

    def test_ckm_beta_gamma(self):
        for order in (None, 0, 1, 2):
            with self.subTest(order=order):
                np_result = ckm_np.ckm_beta_gamma(Vus, Vcb, beta, gamma, delta_expansion_order=order)
                jax_result = ckm_jax.ckm_beta_gamma(Vus, Vcb, beta, gamma, delta_expansion_order=order)
                npt.assert_array_almost_equal(np.array(jax_result), np_result, decimal=12)

    def test_gamma_to_delta(self):
        for order in (None, 0, 1, 2):
            with self.subTest(order=order):
                np_result = ckm_np.gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=order)
                jax_result = ckm_jax.gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=order)
                npt.assert_almost_equal(float(jax_result), np_result, decimal=14)

    def test_beta_gamma_to_delta(self):
        for order in (None, 0, 1, 2):
            with self.subTest(order=order):
                np_result = ckm_np.beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=order)
                jax_result = ckm_jax.beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=order)
                npt.assert_almost_equal(float(jax_result), np_result, decimal=14)

    def test_tree_to_standard(self):
        np_result = ckm_np.tree_to_standard(Vus, Vub, Vcb, gamma)
        jax_result = ckm_jax.tree_to_standard(Vus, Vub, Vcb, gamma)
        npt.assert_array_almost_equal([float(x) for x in jax_result], np_result, decimal=14)

    def test_standard_to_tree(self):
        np_result = ckm_np.standard_to_tree(t12, t13, t23, delta)
        jax_result = ckm_jax.standard_to_tree(t12, t13, t23, delta)
        npt.assert_array_almost_equal([float(x) for x in jax_result], np_result, decimal=14)

    def test_wolfenstein_to_standard(self):
        np_result = ckm_np.wolfenstein_to_standard(laC, A, rhobar, etabar)
        jax_result = ckm_jax.wolfenstein_to_standard(laC, A, rhobar, etabar)
        npt.assert_array_almost_equal([float(x) for x in jax_result], np_result, decimal=14)

    def test_standard_to_wolfenstein(self):
        np_result = ckm_np.standard_to_wolfenstein(t12, t13, t23, delta)
        jax_result = ckm_jax.standard_to_wolfenstein(t12, t13, t23, delta)
        npt.assert_array_almost_equal([float(x) for x in jax_result], np_result, decimal=14)

    def test_tree_to_wolfenstein(self):
        np_result = ckm_np.tree_to_wolfenstein(Vus, Vub, Vcb, gamma)
        jax_result = ckm_jax.tree_to_wolfenstein(Vus, Vub, Vcb, gamma)
        npt.assert_array_almost_equal([float(x) for x in jax_result], np_result, decimal=14)

    def test_wolfenstein_to_tree(self):
        np_result = ckm_np.wolfenstein_to_tree(laC, A, rhobar, etabar)
        jax_result = ckm_jax.wolfenstein_to_tree(laC, A, rhobar, etabar)
        npt.assert_array_almost_equal([float(x) for x in jax_result], np_result, decimal=14)

    def test_msvd(self):
        U_np, S_np, V_np = msvd_np(Mc)
        U_jax, S_jax, V_jax = ckm_jax.msvd(Mc)
        npt.assert_array_almost_equal(np.array(S_jax), S_np, decimal=12)
        # Reconstruction must hold regardless of phase conventions
        npt.assert_array_almost_equal(
            np.array(U_jax) @ np.diag(np.array(S_jax)) @ np.array(V_jax).conj().T,
            Mc, decimal=12
        )


@unittest.skipUnless(JAX_AVAILABLE, "JAX not installed")
class TestJaxGrad(unittest.TestCase):
    """Gradients through CKM functions must be finite and consistent with
    finite differences."""

    def _finite_diff(self, f, x, eps=1e-7):
        return (f(x + eps) - f(x - eps)) / (2 * eps)

    def test_grad_ckm_standard_wrt_delta(self):
        def f(d):
            return jnp.abs(ckm_jax.ckm_standard(t12, t13, t23, d)[0, 2])
        grad = jax.grad(f)(delta)
        fd = self._finite_diff(lambda d: float(jnp.abs(ckm_jax.ckm_standard(t12, t13, t23, d)[0, 2])), delta)
        npt.assert_almost_equal(float(grad), fd, decimal=5)

    def test_grad_gamma_to_delta_wrt_gamma(self):
        f = lambda g: ckm_jax.gamma_to_delta(t12, t13, t23, g)
        grad = jax.grad(f)(gamma)
        fd = self._finite_diff(lambda g: float(ckm_jax.gamma_to_delta(t12, t13, t23, g)), gamma)
        npt.assert_almost_equal(float(grad), fd, decimal=5)

    def test_grad_wolfenstein_to_standard_wrt_etabar(self):
        def f(eb):
            t12_, t13_, t23_, delta_ = ckm_jax.wolfenstein_to_standard(laC, A, rhobar, eb)
            return delta_
        grad = jax.grad(f)(etabar)
        fd = self._finite_diff(
            lambda eb: float(ckm_jax.wolfenstein_to_standard(laC, A, rhobar, eb)[3]),
            etabar
        )
        npt.assert_almost_equal(float(grad), fd, decimal=5)


@unittest.skipUnless(JAX_AVAILABLE, "JAX not installed")
class TestJaxVmap(unittest.TestCase):
    """vmap over a batch of parameter values must match looped evaluation."""

    def test_vmap_ckm_standard(self):
        deltas = jnp.linspace(1.0, 1.5, 5)
        batched = jax.vmap(lambda d: ckm_jax.ckm_standard(t12, t13, t23, d))(deltas)
        for i, d in enumerate(deltas):
            npt.assert_array_almost_equal(
                np.array(batched[i]),
                np.array(ckm_jax.ckm_standard(t12, t13, t23, float(d))),
                decimal=12,
            )

    def test_vmap_gamma_to_delta(self):
        gammas = jnp.linspace(1.0, 1.5, 5)
        batched = jax.vmap(lambda g: ckm_jax.gamma_to_delta(t12, t13, t23, g))(gammas)
        for i, g in enumerate(gammas):
            npt.assert_almost_equal(
                float(batched[i]),
                float(ckm_jax.gamma_to_delta(t12, t13, t23, float(g))),
                decimal=12,
            )
