"""Backend-agnostic implementation of msvd.

``mtakfac`` is not included here because it relies on
``scipy.linalg.fractional_matrix_power``, which has no JAX equivalent.
"""


def _msvd(xp, m):
    """Modified singular value decomposition.

    Returns U, S, V where U†MV = diag(S) and singular values are sorted
    in ascending order (small to large).
    """
    u, s, vdgr = xp.linalg.svd(m)
    order = xp.argsort(s)
    s = s[order]
    u = u[:, order]
    vdgr = vdgr[order]
    return u, s, vdgr.conj().T
