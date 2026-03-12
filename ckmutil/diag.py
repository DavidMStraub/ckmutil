"""Functions for the diagonalization of fermion mass matrices."""

import numpy as np
from scipy.linalg import fractional_matrix_power
from ckmutil._diag_impl import _msvd


def msvd(m):
  """Modified singular value decomposition.

  Returns U, S, V where Udagger M V = diag(S) and the singular values
  are sorted in ascending order (small to large).
  """
  return _msvd(np, m)


def mtakfac(m):
  """Modified Takagi factorization of a (complex) symmetric matrix.

  Returns U, S where U^T M U = diag(S) and the singular values
  are sorted in ascending order (small to large).
  """
  u, s, v = msvd(np.asarray(m, dtype=complex))
  z2 = u.conj().T @ v.conj()
  if np.all(np.abs(z2 - np.diag(np.diag(z2))) < 1e-14): # if z2 is diagonal
    z = np.sqrt(z2)
  else:
    z = fractional_matrix_power(z2,1/2)
  w = v @ z
  return w, s
