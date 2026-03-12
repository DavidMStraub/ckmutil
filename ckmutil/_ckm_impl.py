"""Backend-agnostic implementations of CKM functions.

Each function accepts ``xp`` as its first argument, which should be either
``numpy`` or ``jax.numpy``. This allows the same math to run under both
backends without duplication.

The public API in ``ckmutil.ckm`` calls these with ``xp=numpy``; the JAX API
in ``ckmutil.jax`` calls them with ``xp=jax.numpy`` and leaves any
``jax.jit`` compilation to the caller.
"""


def _ckm_standard(xp, t12, t13, t23, delta):
    c12 = xp.cos(t12)
    c13 = xp.cos(t13)
    c23 = xp.cos(t23)
    s12 = xp.sin(t12)
    s13 = xp.sin(t13)
    s23 = xp.sin(t23)
    eid = xp.exp(1j * delta)
    v = xp.array([
        [c12*c13,
         c13*s12,
         s13/eid],
        [-(c23*s12) - c12*eid*s13*s23,
         c12*c23 - eid*s12*s13*s23,
         c13*s23],
        [-(c12*c23*eid*s13) + s12*s23,
         -(c23*eid*s12*s13) - c12*s23,
         c13*c23],
    ])
    if len(v.shape) > 2:
        v = xp.moveaxis(v, [0, 1], [-2, -1])
    return v


def _gamma_to_delta(xp, t12, t13, t23, gamma, delta_expansion_order=None):
    if delta_expansion_order == 0:
        delta = gamma
    else:
        s13 = xp.sin(t13)
        tan12 = xp.tan(t12)
        tan23 = xp.tan(t23)
        k = s13 * tan23 / tan12
        if delta_expansion_order == 1:
            delta = gamma + k * xp.sin(gamma)
        elif delta_expansion_order == 2:
            delta = gamma + k * xp.sin(gamma) + 1/6 * k**3 * xp.sin(gamma)**3
        elif delta_expansion_order is None:
            delta = xp.arctan(
                (1 - k**2) / (1/xp.tan(gamma) - k * xp.sqrt(1/xp.sin(gamma)**2 - k**2))
            )
        else:
            raise ValueError('delta_expansion_order must be 0, 1, 2, or None.')
    return delta.real


def _beta_gamma_to_delta(xp, beta, gamma, t23, delta_expansion_order=None):
    if delta_expansion_order == 0:
        delta = gamma
    else:
        s23 = xp.sin(t23)
        Rb = xp.sin(beta) / xp.sin(beta + gamma)
        rhobar = Rb * xp.cos(gamma)
        etabar = Rb * xp.sin(gamma)
        if delta_expansion_order == 1:
            delta = gamma + s23**2 * etabar
        elif delta_expansion_order == 2:
            delta = gamma + s23**2 * etabar + s23**4 * rhobar * etabar
        elif delta_expansion_order is None:
            delta = xp.arctan(1 / (1/xp.tan(gamma) - s23**2 * Rb**2 / etabar))
        else:
            raise ValueError('delta_expansion_order must be 0, 1, 2, or None.')
    return delta.real


def _tree_to_standard(xp, Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    s13 = Vub
    c13 = xp.sqrt(1 - s13**2)
    s12 = Vus / c13
    s23 = Vcb / c13
    t13 = xp.arcsin(s13)
    t12 = xp.arcsin(s12)
    t23 = xp.arcsin(s23)
    delta = _gamma_to_delta(xp, t12, t13, t23, gamma, delta_expansion_order)
    return t12.real, t13.real, t23.real, delta


def _standard_to_tree(xp, t12, t13, t23, delta):
    s12 = xp.sin(t12)
    s13 = xp.sin(t13)
    s23 = xp.sin(t23)
    c12 = xp.cos(t12)
    c13 = xp.cos(t13)
    c23 = xp.cos(t23)
    Vus = s12 * c13
    Vub = s13
    Vcb = s23 * c13
    Vcd_complex = -s12*c23 - c12*s23*s13 * xp.exp(1j*delta)
    gamma = xp.angle(-xp.exp(1j*delta) / Vcd_complex)
    return Vus.real, Vub.real, Vcb.real, gamma


def _beta_gamma_to_standard(xp, Vus, Vcb, beta, gamma, delta_expansion_order=None):
    Rb = xp.sin(beta) / xp.sin(beta + gamma)
    rhobar = Rb * xp.cos(gamma)
    a = Vcb**2 * Vus**2 * (1 - Vcb**2) * Rb**2
    b = 1 - Vus**2 - Vcb**2 * (2 * rhobar * (1 - Vus**2) - Vcb**2 * Rb**2)
    c = 2 - Vus**2 - 2 * Vcb**2 * rhobar
    p = (3*b + c**2) / 9
    q = (27*a + 9*b*c + 2*c**3) / 54
    t = 2 * xp.sqrt(p) * xp.sin(xp.arcsin(p**(-3/2) * q) / 3)
    s13 = xp.sqrt(t - c/3)
    c13 = xp.sqrt(1 - s13**2)
    s12 = Vus / c13
    s23 = Vcb / c13
    t13 = xp.arcsin(s13)
    t12 = xp.arcsin(s12)
    t23 = xp.arcsin(s23)
    delta = _beta_gamma_to_delta(xp, beta, gamma, t23, delta_expansion_order)
    return t12.real, t13.real, t23.real, delta


def _standard_to_beta_gamma(xp, t12, t13, t23, delta):
    s12 = xp.sin(t12)
    s13 = xp.sin(t13)
    s23 = xp.sin(t23)
    c12 = xp.cos(t12)
    c13 = xp.cos(t13)
    c23 = xp.cos(t23)
    Vus = s12 * c13
    Vcb = s23 * c13
    Vcd_complex = -s12*c23 - c12*s23*s13 * xp.exp(1j*delta)
    Vtd_complex = s12*s23 - c12*c23*s13 * xp.exp(1j*delta)
    beta = xp.angle(-Vcd_complex / Vtd_complex)
    gamma = xp.angle(-xp.exp(1j*delta) / Vcd_complex)
    return Vus.real, Vcb.real, beta, gamma


def _wolfenstein_to_standard(xp, laC, A, rhobar, etabar):
    rho_plus_i_eta = (
        xp.sqrt(1 - A**2 * laC**4) * (rhobar + 1j*etabar)
        / (xp.sqrt(1 - laC**2) * (1 - A**2 * laC**4 * (rhobar + 1j*etabar)))
    )
    s12 = laC
    s23 = A * laC**2
    s13 = A * laC**3 * xp.abs(rho_plus_i_eta)
    delta = xp.angle(rho_plus_i_eta)
    t12 = xp.arcsin(s12)
    t13 = xp.arcsin(s13)
    t23 = xp.arcsin(s23)
    return t12.real, t13.real, t23.real, delta


def _standard_to_wolfenstein(xp, t12, t13, t23, delta):
    laC = xp.sin(t12)
    A = xp.sin(t23) / laC**2
    rho_plus_i_eta = xp.sin(t13) * xp.exp(1j*delta) / (A * laC**3)
    rhobar_plus_i_etabar = (
        xp.sqrt(1 - laC**2) * rho_plus_i_eta
        / (xp.sqrt(1 - A**2 * laC**4) + xp.sqrt(1 - laC**2) * A**2 * laC**4 * rho_plus_i_eta)
    )
    return laC.real, A.real, rhobar_plus_i_etabar.real, rhobar_plus_i_etabar.imag


def _tree_to_wolfenstein(xp, Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    t12, t13, t23, delta = _tree_to_standard(xp, Vus, Vub, Vcb, gamma, delta_expansion_order)
    return _standard_to_wolfenstein(xp, t12, t13, t23, delta)


def _wolfenstein_to_tree(xp, laC, A, rhobar, etabar):
    t12, t13, t23, delta = _wolfenstein_to_standard(xp, laC, A, rhobar, etabar)
    return _standard_to_tree(xp, t12, t13, t23, delta)


def _ckm_wolfenstein(xp, laC, A, rhobar, etabar):
    t12, t13, t23, delta = _wolfenstein_to_standard(xp, laC, A, rhobar, etabar)
    return _ckm_standard(xp, t12, t13, t23, delta)


def _ckm_tree(xp, Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    t12, t13, t23, delta = _tree_to_standard(xp, Vus, Vub, Vcb, gamma, delta_expansion_order)
    return _ckm_standard(xp, t12, t13, t23, delta)


def _ckm_beta_gamma(xp, Vus, Vcb, beta, gamma, delta_expansion_order=None):
    t12, t13, t23, delta = _beta_gamma_to_standard(xp, Vus, Vcb, beta, gamma, delta_expansion_order)
    return _ckm_standard(xp, t12, t13, t23, delta)
