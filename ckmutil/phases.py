"""Functions for the extraction of mixing angles and phases and for rephasing
general mixing matrices to standard parametrizations."""

import numpy as np
from math import asin, atan, sin, cos, pi
from cmath import phase, exp

def mixing_phases(U):
    """Return the angles and CP phases of the CKM or PMNS matrix
    in standard parametrization, starting from a matrix with arbitrary phase
    convention."""
    f = {}
    # angle t13
    f['t13'] = pi/2 if abs(U[0,2]) >= 1 else asin(abs(U[0,2]))
    s13 = sin(f['t13'])
    c13 = cos(f['t13'])
    if abs(c13) < 1e-7: # special case since U[0,0], U[0,1], U[1,2], U[2,2] are all zero
        # angles t12 and t23
        f['t12'] = pi/2 if abs(U[1,1]) == 0 else atan(abs(U[2,1])/abs(U[1,1]))
        f['t23'] = 0
        s12 = sin(f['t12'])
        c12 = cos(f['t12'])
        # standard phase
        f['delta'] = 0
        # Majorana phases
        f['phi2'] = 0
        f['delta1'] = phase(U[0, 2])
        if abs(s12) < 1e-7:
            f['phi1'] = 0
            f['delta2'] = phase(U[1,1])
            f['delta3'] = phase(-U[2,0])
        elif abs(c12) < 1e-7:
            f['phi1'] = 0
            f['delta2'] = phase(-U[1,0])
            f['delta3'] = phase(-U[2,1])
        else:
            f['delta2'] = phase(U[1,1])
            f['delta3'] = phase(-U[2,1])
            f['phi1'] = 2*phase(-exp(1j*f['delta2'])*U[1,0].conj())
    else:
        # angles t12 and t23
        f['t12'] = pi/2 if abs(U[0,0]) == 0 else atan(abs(U[0,1])/abs(U[0,0]))
        f['t23'] = pi/2 if abs(U[2,2]) == 0 else atan(abs(U[1,2])/abs(U[2,2]))
        s12 = sin(f['t12'])
        c12 = cos(f['t12'])
        s23 = sin(f['t23'])
        c23 = cos(f['t23'])
        # standard phase
        if (
            abs(s12) < 1e-7
            or abs(s13) < 1e-7
            or abs(s23) < 1e-7
            or abs(c12) < 1e-7
            or abs(c23) < 1e-7
        ):
            f['delta'] = 0
        else:
            f['delta'] = -phase((U[0,0].conj()*U[0,2]*U[2,0]*U[2,2].conj()/(c12*c13**2*c23*s13) + c12*c23*s13)/(s12*s23))
        # Majorana phases
        if abs(s12) < 1e-7:
            if abs(s13) < 1e-7:
                f['phi1'] = 0
                f['delta1'] = phase(U[0,0])
            else:
                f['phi1'] = 2*phase(U[0,0].conj()*U[0,2])
                f['delta1'] = phase(U[0,2])
            if abs(s23) < 1e-7:
                f['phi2'] = 0
                f['delta2'] = phase(U[1,1])
                f['delta3'] = phase(U[2,2])
            elif abs(c23) < 1e-7:
                f['phi2'] = 0
                f['delta2'] = phase(U[1,2])
                f['delta3'] = phase(-U[2,1])
            else:
                f['phi2'] = 2*phase(U[1,1].conj()*U[1,2])
                f['delta2'] = phase(U[1,2])
                f['delta3'] = phase(U[2,2])
        elif abs(c12) < 1e-7:
            if abs(s13) < 1e-7:
                f['phi2'] = 0
                f['delta1'] = phase(U[0,1])
            else:
                f['phi2'] = 2*phase(U[0,1].conj()*U[0,2])
                f['delta1'] = phase(U[0,2])
            if abs(s23) < 1e-7:
                f['phi1'] = 0
                f['delta2'] = phase(-U[1,0])
                f['delta3'] = phase(U[2,2])
            elif abs(c23) < 1e-7:
                f['phi1'] = 0
                f['delta2'] = phase(U[1,2])
                f['delta3'] = phase(U[2,0])
            else:
                f['phi1'] = 2*phase(U[2,0].conj()*U[2,2])
                f['delta2'] = phase(U[1,2])
                f['delta3'] = phase(U[2,2])
        elif abs(s13) < 1e-7:
            if abs(s23) < 1e-7:
                f['phi2'] = 0
                f['phi1'] = 2*phase(U[0,0].conj()*U[0,1])
                f['delta1'] = phase(U[0,1])
                f['delta2'] = phase(U[1,1])
                f['delta3'] = phase(U[2,2])
            elif abs(c23) < 1e-7:
                f['phi2'] = 0
                f['phi1'] = 2*phase(U[0,0].conj()*U[0,1])
                f['delta1'] = phase(U[0,1])
                f['delta2'] = phase(U[1,2])
                f['delta3'] = phase(-U[2,1])
            else:
                f['phi1'] = 2*phase(U[2,0].conj()*U[2,2])
                f['phi2'] = 2*phase(U[1,1].conj()*U[1,2])
                f['delta1'] = phase(exp(1j*f['phi1']/2)*U[0,0])
                f['delta2'] = phase(U[1,2])
                f['delta3'] = phase(U[2,2])
        elif abs(s23) < 1e-7:
            f['phi1'] = 2*phase(U[0,0].conj()*U[0,2])
            f['phi2'] = 2*phase(U[0,1].conj()*U[0,2])
            f['delta1'] = phase(U[0,2])
            f['delta2'] = phase(exp(1j*f['phi2']/2)*U[1,1])
            f['delta3'] = phase(U[2,2])
        elif abs(c23) < 1e-7:
            f['phi1'] = 2*phase(U[0,0].conj()*U[0,2])
            f['phi2'] = 2*phase(U[0,1].conj()*U[0,2])
            f['delta1'] = phase(U[0,2])
            f['delta2'] = phase(U[1,2])
            f['delta3'] = phase(exp(1j*f['phi1']/2)*U[2,0])
        else:
            f['delta1'] = phase(exp(1j*f['delta'])*U[0,2])
            f['delta2'] = phase(U[1,2])
            f['delta3'] = phase(U[2,2])
            f['phi1'] = 2*phase(exp(1j*f['delta1'])*U[0,0].conj())
            f['phi2'] = 2*phase(exp(1j*f['delta1'])*U[0,1].conj())
    return f

def rephase_standard(UuL, UdL, UuR, UdR):
    """Function to rephase the quark rotation matrices in order to
    obtain the CKM matrix in standard parametrization.

    The input matrices are assumed to diagonalize the up-type and down-type
    quark matrices like

    ```
    UuL.conj().T @ Mu @ UuR = Mu_diag
    UdL.conj().T @ Md @ UdR = Md_diag
    ```

    The CKM matrix is given as `VCKM = UuL.conj().T @ UdL`.

    Returns a tuple with the rephased versions of the input matrices.
    """
    K = UuL.conj().T @ UdL
    f = mixing_phases(K)
    Fdelta = np.diag(np.exp([1j*f['delta1'], 1j*f['delta2'], 1j*f['delta3']]))
    Fphi = np.diag(np.exp([-1j*f['phi1']/2., -1j*f['phi2']/2., 0]))
    return UuL @ Fdelta, UdL @ Fphi.conj(), UuR @ Fdelta, UdR @ Fphi.conj()

def rephase_pmns_standard(Unu, UeL, UeR):
    """Function to rephase the lepton rotation matrices in order to
    obtain the PMNS matrix in standard parametrization.

    The input matrices are assumed to diagonalize the charged lepton and
    neutrino mass matrices like

    ```
    UeL.conj().T @ Me @ UeR = Me_diag
    Unu.T @ Mnu @ Unu = Mnu_diag
    ```

    The PMNS matrix is given as `UPMNS = UeL.conj().T @ Unu`.

    Returns a tuple with the rephased versions of the input matrices.
    """
    U = UeL.conj().T @ Unu
    f = mixing_phases(U)
    Fdelta = np.diag(np.exp([1j*f['delta1'], 1j*f['delta2'], 1j*f['delta3']]))
    return Unu, UeL @ Fdelta, UeR @ Fdelta
