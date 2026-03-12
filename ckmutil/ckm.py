"""Functions needed for the CKM quark mixing matrix."""

import numpy as np
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


def ckm_standard(t12, t13, t23, delta):
    r"""CKM matrix in the standard parametrization and standard phase
    convention.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta=\gamma$ in radians

    Returns
    -------
    - `v`: CKM matrix in the standard parametrization and standard phase
        convention
    """
    return _ckm_standard(np, t12, t13, t23, delta)


def gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=None):
    r"""CKM phase $\delta$ in terms of $\gamma$.

    By default, no analytical approximation is made for the CKM phase
    $\delta$ but the optional argument `delta_expansion_order` allows to use
    the very accurate analytical approximation $\delta=\gamma$ for
    `delta_expansion_order=0` or to include higher-order corrections to this
    approximation for `delta_expansion_order=1` or `delta_expansion_order=2`.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `delta`: CKM phase $\delta$ in radians
    """
    return _gamma_to_delta(np, t12, t13, t23, gamma, delta_expansion_order)


def beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=None):
    r"""CKM phase $\delta$ in terms of $\beta$ and $\gamma$.

    By default, no analytical approximation is made for the CKM phase
    $\delta$ but the optional argument `delta_expansion_order` allows to use
    the very accurate analytical approximation $\delta=\gamma$ for
    `delta_expansion_order=0` or to include higher-order corrections to this
    approximation for `delta_expansion_order=1` or `delta_expansion_order=2`.

    Parameters
    ----------
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-6})$.

    Returns
    -------
    - `delta`: CKM phase $\delta$ in radians
    """
    return _beta_gamma_to_delta(np, beta, gamma, t23, delta_expansion_order)


def tree_to_standard(Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    r"""Function to convert from the CKM matrix in the tree parametrization to
    the CKM matrix in the standard parametrization.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians
    """
    return _tree_to_standard(np, Vus, Vub, Vcb, gamma, delta_expansion_order)


def standard_to_tree(t12, t13, t23, delta):
    r"""Function to convert from the CKM matrix in the standard parametrization
    to the CKM matrix in the tree parametrization.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Returns
    -------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians
    """
    return _standard_to_tree(np, t12, t13, t23, delta)


def beta_gamma_to_standard(Vus, Vcb, beta, gamma, delta_expansion_order=None):
    r"""Function to convert from the CKM matrix in the beta-gamma parametrization
    to the CKM matrix in the standard parametrization.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-6})$.

    Returns
    -------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians
    """
    return _beta_gamma_to_standard(np, Vus, Vcb, beta, gamma, delta_expansion_order)


def standard_to_beta_gamma(t12, t13, t23, delta):
    r"""Function to convert from the CKM matrix in the standard parametrization
    to the CKM matrix in the beta-gamma parametrization.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Returns
    -------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians
    """
    return _standard_to_beta_gamma(np, t12, t13, t23, delta)


def wolfenstein_to_standard(laC, A, rhobar, etabar):
    r"""Function to convert from the CKM matrix in the Wolfenstein parametrization
    to the CKM matrix in the standard parametrization.

    Parameters
    ----------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Returns
    -------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    return _wolfenstein_to_standard(np, laC, A, rhobar, etabar)


def standard_to_wolfenstein(t12, t13, t23, delta):
    r"""Function to convert from the CKM matrix in the standard parametrization
    to the CKM matrix in the Wolfenstein parametrization.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Returns
    -------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    return _standard_to_wolfenstein(np, t12, t13, t23, delta)


def tree_to_wolfenstein(Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    r"""Function to convert from the CKM matrix in the tree parametrization to
    the CKM matrix in the Wolfenstein parametrization.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    return _tree_to_wolfenstein(np, Vus, Vub, Vcb, gamma, delta_expansion_order)


def wolfenstein_to_tree(laC, A, rhobar, etabar):
    r"""Function to convert from the CKM matrix in the Wolfenstein parametrization
    to the CKM matrix in the tree parametrization.

    Parameters
    ----------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Returns
    -------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    return _wolfenstein_to_tree(np, laC, A, rhobar, etabar)


def ckm_wolfenstein(laC, A, rhobar, etabar):
    r"""CKM matrix in the Wolfenstein parametrization and standard phase
    convention.

    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$

    Parameters
    ----------

    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Returns
    -------
    - `v`: CKM matrix in the Wolfenstein parametrization and standard phase
        convention
    """
    return _ckm_wolfenstein(np, laC, A, rhobar, etabar)


def ckm_tree(Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    r"""CKM matrix in the tree parametrization and standard phase
    convention.

    In this parametrization, the parameters are directly measured from
    tree-level $B$ decays. It is thus particularly suited for new physics
    analyses because the tree-level decays should be dominated by the Standard
    Model. By default, no analytical approximation is made for the CKM phase
    $\delta$ but the optional argument `delta_expansion_order` allows to use
    the very accurate analytical approximation $\delta=\gamma$ for
    `delta_expansion_order=0` or to include higher-order corrections to this
    approximation for `delta_expansion_order=1` or `delta_expansion_order=2`.

    Relation to the standard parametrization:

    - $V_{us} = \cos \theta_{13} \sin \theta_{12}$
    - $|V_{ub}| = \sin \theta_{13}$
    - $V_{cb} = \cos \theta_{13} \sin \theta_{23}$
    - $\delta = \gamma + \mathcal{O}(10^{-3})$

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: CKM phase $\gamma=\delta$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `v`: CKM matrix in the tree parametrization and standard phase
    """
    return _ckm_tree(np, Vus, Vub, Vcb, gamma, delta_expansion_order)


def ckm_beta_gamma(Vus, Vcb, beta, gamma, delta_expansion_order=None):
    r"""CKM matrix in the beta-gamma parametrization and standard phase
    convention.

    In this parametrization, the two angles $\beta$ and $\gamma$ of the
    Unitarity Triangle are used as input parameters in addition to the CKM
    matrix elements $V_{us}$ and $V_{cb}$. By default, no analytical
    approximation is made for the CKM phase $\delta$ but the optional argument
    `delta_expansion_order` allows to use the very accurate analytical
    approximation $\delta=\gamma$ for `delta_expansion_order=0` or to include
    higher-order corrections to this approximation for
    `delta_expansion_order=1` or `delta_expansion_order=2`.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-6})$.

    Returns
    -------
    - `v`: CKM matrix in the beta-gamma parametrization and standard phase
    """
    return _ckm_beta_gamma(np, Vus, Vcb, beta, gamma, delta_expansion_order)
