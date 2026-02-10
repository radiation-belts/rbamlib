import numpy as np
import rbamlib.constants
import rbamlib.models.dip.T as T


def Y(al):
    """
    Approximation of the integral function Y related to the second adiabatic invariant, derived in the dipole approximation.

    Parameters
    ----------
    al : float or ndarray
         Equatorial pitch angle, in radians.

    Returns
    -------
    float or ndarray
         Value of Y

    Notes
    -----
        See :cite:t:`schulz:1974`.

    .. math::
        Y( \\alpha ) \\approx 2(1 - \\sin( \\alpha ))T_0 + (T_0 - T_1) \\cdot \\left( \\sin( \\alpha ) \\cdot \\ln( \\sin( \\alpha ) ) + 2 \\sin( \\alpha ) - 2 \\sqrt{ \\sin( \\alpha ) } \\right)
        Y(0) = 2 \\cdot T(0)
    """
    T0 = rbamlib.constants.T0
    T1 = rbamlib.constants.T1

    # Ensure input is a numpy array for vectorized operations.
    al_arr = np.atleast_1d(al)
    y = np.sin(al_arr)

    # Compute the components of the expression. Suppress warnings for log(0) or invalid values.
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_term = y * np.log(y) + 2 * y - 2 * np.sqrt(y)

    Y = 2 * (1 - y) * T0 + (T0 - T1) * computed_term

    # For angles where al == 0, the analytical expression is Y(0) = 2 * T(0)
    mask = (al_arr == 0)
    Y[mask] = 2 * T(0)

    # Return a scalar if the input was a scalar.
    if np.isscalar(al) or (hasattr(al, "ndim") and al.ndim == 0):
        return Y[0]
    return Y