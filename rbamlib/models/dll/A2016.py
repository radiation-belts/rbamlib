import numpy as np


def A2016(L, kp, dll_type='M'):
    r"""
    Calculate radial diffusion coefficient following Ali et al. :cite:yearpar:`ali:2016` model.

    Parameters
    ----------
    L : ndarray
        Array of L values, representing the radial distance in Earth radii where magnetic field lines cross the magnetic equator.
    kp : ndarray
        Kp-index, vector of geomagnetic activity indices corresponding to the times.
    dll_type : str, default='E'
        A string, indicating which diffusion coefficient to calculate.
        'M' or 'B' stands for magnetic radial diffusion coefficient (eq. 14).
        'E' stands for electric diffusion coefficient (eq. 15).
        'ME', 'BE', 'both', or '' computes both coefficients and returns a tuple (`dllb`, `dlle`).

    Returns
    -------
    dllb : numpy.ndarray
        Magnetic radial diffusion coefficient, in 1/days (if dll_type='M', 'B').
    dlle : numpy.ndarray
        Electric radial diffusion coefficient, in 1/days (if dll_type='E').
    dllb, dlle : tuple of numpy.ndarray
        Both magnetic and electric diffusion coefficients (dllb, dlle) if dll_type='ME', 'BE', 'both', or ''.

    Notes
    -----
    The magnetic radial diffusion coefficient is calculated as:

    .. math::
        D^{B}_{LL} = \exp \left( a_1 + b_1 \cdot Kp \cdot L^* + L^* \right)

    The electric radial diffusion coefficient is calculated as:

    .. math::
        D^{E}_{LL} = \exp \left( a_2 + b_2 \cdot Kp \cdot L^* + c_2 \cdot L^* \right)

    where the constants are given by:

    .. math::
        a_1 = -16.253, \quad b_1 = 0.224, \quad  a_2 = -16.951, \quad b_2 = 0.181, \quad c_2 = 1.982
    """
    dllb, dlle = None, None

    if dll_type in {'M', 'm', 'B', 'b', '', 'ME', 'me', 'BE', 'be', 'both'}:
        dllb = np.exp(-16.253 + 0.224 * kp * L + L)

    if dll_type in {'E', 'e', '', 'ME', 'me', 'BE', 'be', 'both'}:
        dlle = np.exp(-16.951 + 0.181 * kp * L + 1.982 * L)

    if dll_type in {'M', 'm', 'B', 'b'}:
        return dllb
    elif dll_type in {'E', 'e'}:
        return dlle
    elif dll_type in {'ME', 'me', 'BE', 'be', 'both', ''}:
        return dllb, dlle