import numpy as np

def D2002(r, ne_eq, L=None, Rmax=None, alpha=None):
    r"""
    Electron density along a field line following Denton et al. (2002) [#]_ eq. (1, 2, 3, 4).

    Parameters
    ----------
    r : array_like
        Geocentric distance along the field line in Earth radii.
    ne_eq : array_like
        Equatorial electron density :math:`n_{e,eq}` in cm⁻³.
    L : array_like, optional
        McIlwain L-shell. Provide `L` if you do not pass `Rmax`.
    Rmax : array_like, optional
        Maximum geocentric distance along the field line in Earth radii.
        If given, it is used directly in the power law and in :math:`\alpha`.
    alpha : array_like, optional
        Power-law exponent controlling the density variation :math:`\alpha`.
        This parameter can be used to drive other model of similar form.

    Returns
    -------
    ne : ndarray
        Electron density :math:`n_e` in cm⁻³.

    Notes
    -----
    - The original data set used to derive these relations was limited to
      roughly :math:`2 \le n_{e,eq} \le 1500\ \text{cm}^{-3}` and
      :math:`2 \lesssim L \lesssim 8.5`.

    The density varies with geocentric distance :math:`r` (along a field line)
    as a power law of the equatorial value :math:`n_{e,eq}` (eq. 1):

    .. math::
        n_e(r) = n_{e,eq} \left( \frac{R_{\max}}{r} \right)^{\alpha} = n_{e,eq} \left( L \cdot \frac{1}{r} \right)^{\alpha}

    where :math:`R_{\max}\!\approx L\,R_{Earth}` is the maximum geocentric distance
    on the field line (near the magnetic equator), considering that `r` is in units of Earth radii.
    Denton et al. parameterized the exponent :math:`\alpha` as (eq. 2-4) as:

    .. math::
        \alpha = 8 - 3\log_{10}(n_{e,eq}) + 0.28[\log_{10}(n_{e,eq})]^2 - 0.43\frac{R_{\max}}{R_{Earth}}

    which becomes (using :math:`R_{\max}/R_{Earth} \approx L`) the commonly used form

    .. math::
        \alpha = 8 - 0.43L - 3\log_{10}(n_{e,eq}) + 0.28[\log_{10}(n_{e,eq})]^2

    References
    ----------
    .. [#] Denton, R. E., Goldstein, J., & Menietti, J. D. (2002), Field line dependence of magnetospheric electron density, Geophys. Res. Lett., 29(24), 2205. https://doi.org/10.1029/2002GL015963
    """
    r = np.asarray(r, dtype=float)
    ne_eq = np.asarray(ne_eq, dtype=float)

    if Rmax is None:
        if L is None:
            raise ValueError("Provide either Rmax (in Re) or L.")
        L = np.asarray(L, dtype=float)
        Rmax_eff = L  # since inputs are in Re, Rmax/Re = L
    else:
        Rmax_eff = np.asarray(Rmax, dtype=float)

    if alpha is None:
        log10_ne = np.log10(ne_eq)
        if Rmax is None:
            alpha = 8.0 - 0.43 * L - 3.0 * log10_ne + 0.28 * (log10_ne ** 2)
        else:
            alpha = 8.0 - 3.0 * log10_ne + 0.28 * (log10_ne ** 2) - 0.43 * Rmax_eff
    else:
        alpha = np.asarray(alpha, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        ne = ne_eq * np.power(Rmax_eff / r, alpha)

    return ne
