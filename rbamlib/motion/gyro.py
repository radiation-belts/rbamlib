import numpy as np
import rbamlib.constants
from rbamlib.conv import en2gamma

def f_gyro(B, en=None, q=rbamlib.constants.q, m=rbamlib.constants.me):
    r"""
    Calculate the oscillation gyro (cyclotron) frequency.

    Parameters
    ----------
    B : float or ndarray
        Magnetic field strength in Gauss.
    en : float or ndarray, optional
        Kinetic energy in MeV. If provided, the relativistic correction is applied.
    q : float or ndarray, optional
        Particle charge (SGS charge).
    m : float or ndarray, optional
        Particle mass (g).
    
    Returns
    -------
    float or ndarray
        Gyro frequency in Hz.
    
    Notes
    -----
    For a non-relativistic particle, the gyro frequency is given by

        f = |q| B / (2π m c)

    For a relativistic particle with kinetic energy (K) the Lorentz factor is

        γ = 1 + K/(mc²)

    and the gyro frequency becomes

        f = |q| B / (2π γ m c)
        
    Here, mc² (rest mass energy) is provided by rbamlib.constants.MC2 and is in MeV.
    """
    # Compute gamma if energy is provided (en is in MeV)
    gamma = 1.0
    if en is not None:
        gamma = en2gamma(en)
    
    # Calculate non default option for particles other than electrons
    if m != rbamlib.constants.me:
        erg_to_MeV = 1e-7 / 1.60218e-13  # ≈ 6.24151e5 MeV per erg
        mc2_erg = m * rbamlib.constants.c**2
        # Calculate mc2 in MeV
        mc2 = mc2_erg * erg_to_MeV
        gamma = en2gamma(en, mc2)

    # Compute f
    f = np.abs(q) * B / (2 * np.pi * gamma * m * rbamlib.constants.c)

    return f       


def omega_gyro(B, en=None, q=rbamlib.constants.q, m=rbamlib.constants.me):
    """
        Calculate the angular (ω) gyro (cyclotron) frequency.        
    """
    return 2 * np.pi * f_gyro(B, en, q, m)


def T_gyro(B, en=None, q=rbamlib.constants.q, m=rbamlib.constants.me):
    """
        Calculate gyto period        
    """
    return 2 * np.pi / omega_gyro(B, en, q, m)

