import numpy as np

def O2014(L, kp, dll_type='E'):
    r"""
    Calculate radial diffusion coefficient following Ozeke et al. (2014) [#]_ model.

    Parameters
    ----------
    kp : ndarray
        Kp-index, vector of geomagnetic activity indices corresponding to the times.
    L : ndarray
        Array of L values, representing the radial distance in Earth radii where magnetic field lines cross the magnetic equator.
    dll_type : str, default='E'
        A string, indicating which diffusion coefficient to calculate.
        'M' or 'B': stands for magnetic radial diffusion coefficient (eq. 20).
        'E' stands for electric diffusion coefficient (eq. 23).
        'ME', 'BE', 'both', or '' computes both coefficients and returns a tuple (`dllb`, `dlle`).


    Returns
    -------
    dllb : numpy.ndarray
        Magnetic radial diffusion coefficient, in 1/days (if dll_type='B').
    dlle : numpy.ndarray
        Electric radial diffusion coefficient, in 1/days (if dll_type='E').
    dllb, dlle : tuple of numpy.ndarray
        Both electromagnetic and electrostatic diffusion coefficients (dllb, dlle) if dll_type='ME' or ''.

    Notes
    -----
    The electromagnetic radial diffusion coefficient is calculated as:

    .. math::
        D^{B}_{LL} = 6.62 \times 10^{-13} \cdot L^8 10^{-0.0327 L^2 + 0.625 L - 0.0108 Kp^2 + 0.499 Kp}

    The electrostatic radial diffusion coefficient is calculated as:

    .. math::
        D^{E}_{LL} = 2.16 \times 10^{-8} \cdot L^6 \cdot 10^{0.217 L + 0.461 Kp}

    References
    ----------
    .. [#] Ozeke, L. G., Mann, I. R., Murphy, K. R., Jonathan Rae, I., & Milling, D. K. (2014). Analytic expressions for ULF wave radiation belt radial diffusion coefficients. Journal of Geophysical Research, [Space Physics], 119(3), 1587–1605. https://doi.org/10.1002/2013JA019204
    """
    dllb, dlle = None, None

    if dll_type in {'M', 'm', '', 'ME', 'me', 'both', 'B', 'b', 'BE', 'be'}:
        dllb = 6.62e-13 * L ** 8 * 10 ** (-0.0327 * L ** 2 + 0.625 * L - 0.0108 * kp ** 2 + 0.499 * kp)

    if dll_type in {'E', 'e', '', 'ME', 'me', 'both', 'BE', 'be'}:
        dlle = 2.16e-8 * L ** 6 * 10 ** (0.217 * L + 0.461 * kp)

    if dll_type in {'M', 'm', 'B', 'b'}:
        return dllb
    elif dll_type in {'E', 'e'}:
        return dlle
    elif dll_type in {'ME', 'me', 'both', '', 'BE', 'be'}:
        return dllb, dlle


