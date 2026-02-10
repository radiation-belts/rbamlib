import numpy as np


def L2016(L, kp, mu, dll_type='E'):
    r"""
    Calculate radial diffusion coefficient following Liu et al. :cite:yearpar:`liu:2016` model.

    Parameters
    ----------
    L : ndarray
        Array of L values, representing the radial distance in Earth radii where magnetic field lines cross the magnetic equator.
    kp : ndarray
        Kp-index, vector of geomagnetic activity indices corresponding to the times.
    mu : ndarray
        First adiabatic invariant in MeV/G.
    dll_type : str, default='E'
        This function only calculates the electric radial diffusion coefficient ('E').

    Returns
    -------
    dlle : numpy.ndarray
        Electric radial diffusion coefficient, in 1/days.

    Notes
    -----
    The electric radial diffusion coefficient is calculated as:

    .. math::
        D^{E}_{LL} = 1.115 \times 10^{-6} \cdot 10^{0.281 \cdot Kp} \cdot L^{8.184} \cdot \mu^{-0.608}
    """
    if dll_type not in {'E', 'e'}:
        raise ValueError("This function only computes electric radial diffusion (dll_type='E').")

    dlle = 1.115e-6 * 10 ** (0.281 * kp) * L ** 8.184 * mu ** -0.608

    return dlle
