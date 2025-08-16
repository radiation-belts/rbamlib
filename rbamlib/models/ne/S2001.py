import numpy as np


def S2001(L, Lpp=None, kp=0.0, mask_invalid=True):
    r"""
    Equatorial electron density model of the plasmasphere and plasmatrough
    from Sheeley et al. (2001) [#]_.

    Parameters
    ----------
    L : float or ndarray
        McIlwain L-shell. Validated range: :math:`3 \le L \le 7`.
    Lpp : float or ndarray, optional
        Plasmapause location in L. If supplied, :math:`L \le L_{\mathrm{pp}}`
        uses the plasmasphere model and :math:`L > L_{\mathrm{pp}}` uses the trough
        model. If ``None``, all inputs default to the **plasmasphere** model.
    kp : float or ndarray, optional
        Kp index. User for plasmatrough calculation. Default is 0
    mask_invalid : bool, optional
        If ``True`` (default), values outside :math:`3 \le L \le 7` are returned
        as ``NaN``. If ``False``, equations are evaluated without restriction.

    Returns
    -------
    ne : float or ndarray
        Equatorial electron density, :math:`\mathrm{cm}^{-3}`.

    Notes
    -----
    **Plasmasphere (Sheeley et al. 2001, Eq. 6):**

    .. math::
        n_e^{\mathrm{ps}}(L)
        = 1390 \left(\frac{3}{L}\right)^{4.83}, \quad 3 \le L \le 7.

    **Plasmatrough (Sheeley et al. 2001, Eq. 7):**

    .. math::
        n_e^{\mathrm{tr}}(L,\mathrm{LT})
        = 124 \left(\frac{3}{L}\right)^{4.0}
          + 36 \left(\frac{3}{L}\right)^{3.5}
            \cos\!\left(
              \frac{\pi}{12}\Big[\mathrm{LT}-\big(7.7(3/L)^{2.0}+12\big)\Big]
            \right).

        \mathrm{LT} = 0.145\,K_p^2 - 2.63\,K_p + 21.86

    References
    ----------
    .. [#] Sheeley, B. W., Moldwin, M. B., Rassoul, H. K., & Anderson, R. R. (2001). An empirical plasmasphere and trough density model: CRRES observations. Journal of Geophysical Research, [Space Physics], 106(A11), 25631–25641. https://doi.org/10.1029/2000JA000286
    """
    L = np.asarray(L, dtype=float)

    if Lpp is None:
        ne = _S2001_plasmasphere(L)
    else:
        Lpp = np.asarray(Lpp, dtype=float)
        use_ps = L <= Lpp
        LT = _S2001_LT(kp)
        ne = np.where(use_ps, _S2001_plasmasphere(L), _S2001_trough(L, LT))

    if mask_invalid:
        valid = (L >= 3.0) & (L <= 7.0)
        ne = np.where(valid, ne, np.nan)

    return ne


# ----------------------------------------------------------------------
# Supporting helper functions
# ----------------------------------------------------------------------

def _S2001_LT(kp):
    r"""
    Local time of maximum trough density as a function of Kp (Gallagher et al., 1998) [#]_.

    Parameters
    ----------
    kp : float or ndarray
        Kp index.

    Returns
    -------
    LT : float or ndarray
        Local time of maximum density in hours (hours).

    Notes
    -----
    Empirical quadratic relation between Kp and local time (LT) found from GEOS-2
    observations at geosynchronous orbit (L = 6.6).

    .. math::
        \mathrm{LT} = 0.145\,K_p^2 - 2.63\,K_p + 21.86

    References
    ----------
    .. [#] Gallagher, D. L., et al., 1998,
    """
    kp = np.asarray(kp, dtype=float)
    LT = 0.145 * kp ** 2 - 2.63 * kp + 21.86

    # Preserve scalar return for scalar input
    if LT.shape == ():
        return float(LT)
    return LT


def _S2001_plasmasphere(L):
    r"""
    Plasmasphere electron density model (Sheeley et al. 2001, Eq. 6).

    .. math::
        n_e(L) = 1390 \left(\frac{3}{L}\right)^{4.83}.
    """
    L = np.asarray(L, dtype=float)
    x = 3.0 / L
    return 1390.0 * x ** 4.83


def _S2001_trough(L, LT):
    r"""
    Plasmatrough electron density model (Sheeley et al. 2001, Eq. 7).

    .. math::
        n_e(L,\mathrm{LT}) =
        124 \left(\frac{3}{L}\right)^{4.0}
        + 36 \left(\frac{3}{L}\right)^{3.5}
        \cos\!\left(\frac{\pi}{12} \,[\mathrm{LT}-(7.7(3/L)^{2}+12)]\right).
    """
    L = np.asarray(L, dtype=float)
    LT = np.asarray(LT, dtype=float)
    x = 3.0 / L
    phase = (LT - (7.7 * x ** 2.0 + 12.0)) * (np.pi / 12.0)
    return 124.0 * x ** 4.0 + 36.0 * x ** 3.5 * np.cos(phase)
