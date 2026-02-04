import numpy as np

def D2004(L, Lpp=None, R13=13.0, a=None):
    r"""
    Denton et al. :cite:yearpar:`denton:2004` -style plasmaspheric electron density (equatorial or along field line).

    The plasmasphere (``L <= Lpp`` or when no ``Lpp`` is provided) uses
    Denton et al. (2004) equation (5). The plasmatrough (``L > Lpp``) uses a power-law form (eq. 11).

    Parameters
    ----------
    L : float or ndarray
        McIlwain L-shell.
    Lpp : float or ndarray, optional
        Plasmapause location. If supplied, :math:`L \le L_{\mathrm{pp}}`
        uses the plasmasphere model and :math:`L > L_{\mathrm{pp}}` uses the trough
        model. If ``None``, all inputs default to the plasmasphere model.
    R13 : float, optional
        13-month smoothed sunspot number. If is not ``None``, solar–cycle term is added in :math:`\log_{10} n_e`.
        The default value 13 corresponds to the Denton et al. (2004).
    a : tuple of 4 floats or ``None``, optional
        Coefficients :math:`(a_1,a_2,a_3,a_4)`:

        - :math:`a_1, a_2` for plasmasphere (eq. 5).
        - :math:`a_3, a_4` for plasmatrough (eq. 11).

        If ``None``, defaults are used: (3.78, -0.324, 3.77, -3.45).

    Returns
    -------
    ne : float or ndarray
        Electron density in cm⁻³.

    Notes
    -----

    ----

    **Plasmasphere** (:math:`L \le L_{pp}`, Eq. 5)

    The sunspot term uses 13-month smoothed sunspot number :math:`\text{R13}=\overline{R}`.
    If ``R13`` is ``None`` the sunspot term is omitted.

    .. math::

       \log_{10}(n_e) = a_1 + a_2 L
       + (0.00127 \overline{R} - 0.0635) \exp(-\tfrac{L-2}{1.5})


    ----

    **Plasmatrough** (:math:`L > L_{pp}`, Eq. 11)

    .. math::

       n_e = a_3 * L ^ {a_4}

    ----
    """
    L = np.asarray(L, dtype=float)

    if a is None:
        # Default coefficients from Denton et al. (2004)
        a1, a2, a3, a4 = 3.78, -0.324, 3.77, -3.45
    else:
        if len(a) != 4:
            raise ValueError("`a` must be a 4-tuple/list: (a1, a2, a3, a4).")
        a1, a2, a3, a4 = a

    if Lpp is None:
        return D2004_plasmasphere(L, a1, a2, R13)
    else:
        Lpp = np.asarray(Lpp, dtype=float)
        use_ps = L <= Lpp
        return np.where(
            use_ps,
            D2004_plasmasphere(L, a1, a2, R13),
            D2004_trough(L, a3, a4)
        )


def D2004_plasmasphere(L, a1, a2, R13=None):
    r"""
       Plasmaspheric density model of Denton et al. (2004), Eq. (5).

       .. math::

           \log_{10}(n_e}) = a_1 + a_2 L
           + \left(0.00127 R_{13} - 0.0635\right) \exp(-\tfrac{L-2}{1.5})


       Parameters
       ----------
       L : float or ndarray
           McIlwain L-shell.
       a1, a2 : float
           Linear fit coefficients.
       R13 : float, optional
           13-month smoothed sunspot number. If ``None`` the sunspot term is omitted.

       Returns
       -------
       ne : float or ndarray
           Electron density, cm⁻³.
       """
    L = np.asarray(L, dtype=float)
    log10_ps = a1 + a2 * L
    if R13 is not None:
        log10_ps += (0.00127 * float(R13) - 0.0635) * np.exp(-(L - 2.0) / 1.5)
    return 10.0 ** log10_ps


def D2004_trough(L, a3, a4):
    r"""
    Plasmatrough density model of Denton et al. (2004), Eq. (11).

    .. math::

       n_e = a_3 * L ^ {a_4}

    Parameters
    ----------
    L : float or ndarray
        McIlwain L-shell.
    a3, a4 : float
        Power-law coefficients.

    Returns
    -------
    ne : float or ndarray
        Electron density, cm⁻³.
    """
    L = np.asarray(L, dtype=float)

    return a3 * L ** a4
