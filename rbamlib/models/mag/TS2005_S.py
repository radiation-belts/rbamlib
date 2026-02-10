import numpy as np


def TS2005_S(Nsw, Vsw, Bz, fillval=None):
    r"""
    Compute the Tsyganenko & Sitnov (2005) :cite:yearpar:`tsyganenko:2005` source functions :math:`S_k`, per Eq. (8):

    .. math::
        S_k = \left(\frac{N_{\mathrm{sw}}}{5}\right)^{\lambda_k}
              \left(\frac{V_{\mathrm{sw}}}{400}\right)^{\beta_k}
              \left(\frac{B_s}{5}\right)^{\gamma_k},

    where :math:`B_s = -B_z` if :math:`B_z < 0` else 0, and
    :math:`(\lambda_k, \beta_k, \gamma_k)` are from the Tsyganenko & Sitnov (2005) model tables.

    Parameters
    ----------
    Nsw : 1D ndarray
        Solar wind density array.
    Vsw : 1D ndarray
        Solar wind velocity array.
    Bz : 1D ndarray
        IMF Bz array.
    fillval: float, optional
        Fill values of S if is computed as np.nan (due to invalid input)


    Returns
    -------
    S : 2D ndarray, shape (N, 6)
        6 different :math:`S_k(t)` time series (one column per k).
    """
    # Hard-coded exponents:
    lambda_ = np.array([0.39, 0.46, 0.39, 0.42, 0.41, 1.29])
    beta_ = np.array([0.80, 0.18, 2.32, 1.25, 1.60, 2.40])
    gamma_ = np.array([0.87, 0.67, 1.32, 1.29, 0.69, 0.53])

    Bs = np.where(Bz < 0, -Bz, 0.0)
    # S_k(t) for each k = 1..6, broadcast across time
    S = ((Nsw / 5.0) ** lambda_[:, None]
         * (Vsw / 400.0) ** beta_[:, None]
         * (Bs / 5.0) ** gamma_[:, None])

    if fillval is not None:
        S[np.isnan(S)] = fillval

    return S.T
