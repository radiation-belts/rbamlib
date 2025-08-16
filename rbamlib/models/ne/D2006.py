import numpy as np


def D2006(L, R: float = 78.0):
    r"""
    Calculates electron density following Denton et al. (206) [#]_ model.
    L-only variant in cm^-3.


    .. warning::
       This function under development. Implementation will be added soon.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    .. math::
     n_e = 10 ^ ( -0.324*L + 3.78 + (0.00127*R - 0.0635) * exp((2 - L)/1.5) )

             n_e(L) = n_e(L_{pp}) \times 10^{\,(L - L_{pp})/s(t)},

    References
    ----------
    .. [#] Denton, R. E., Wang, Y., Webb, P. A., Tengdin, P. M., Goldstein, J., Redfern, J. A., & Reinisch, B. W. (2012). Magnetospheric electron density long-term (>1 day) refilling rates inferred from passive radio emissions measured by IMAGE RPI during geomagnetically quiet times: MAGNETOSPHERIC ELECTRON DENSITY REFILLING RATES. Journal of Geophysical Research, 117(A3). https://doi.org/10.1029/2011JA017274
    """
    L = np.asarray(L, dtype=float)
    ne = 10.0 ** (
        -0.324 * L + 3.78 + (0.00127 * R - 0.0635) * np.exp((2.0 - L) / 1.5)
    )
    return ne
