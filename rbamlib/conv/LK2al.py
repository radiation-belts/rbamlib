import numpy as np
from scipy.optimize import root_scalar
import rbamlib.models.dip
import rbamlib.constants


def LK2al(L, K=0, B0=rbamlib.models.dip.B0, xtol=1e-9):
    r"""
    Calculate the pitch-angle in radians given the L-shell and the adiabatic invariant K.

    This function uses numerical methods to find the pitch angle where the calculated function based on the dipole field properties matches K.

    Parameters
    ----------
    L : float or ndarray
        L-shell value at which the pitch-angle is to be calculated.
    K : float or ndarray, default K=0
        Adiabatic invariant K, in units of G^0.5 * R. Should be the same shape as L if an array is provided. At default K=0 pitch angle is 90 degrees.
    B0 : function, default=rbamlib.models.dip.B0
        Function that calculates the magnetic field in Gauss at a given L-shell, or `r`. By default, Earth's dipole field model is used.
    xtol : float, default=1e-9
        Tolerance for the numerical root finder.

    Returns
    -------
    ndarray
        Pitch angle in radians.

    Notes
    -----
    The pitch-angle :math:`\alpha` is calculated by solving the equation:

    .. math::
        \frac{Y(\alpha)}{\sin(\alpha)} = \frac{K}{R \sqrt{B_0(L)}}

    where:
        - :math:`Y(\alpha)` is a Y-function specific to the dipole field,
        - :math:`B_0(L)` is the dipole magnetic field strength at L-shell `L` (or `r`).

    The function uses the `fsolve` method from SciPy to numerically solve for :math:`\alpha` at each value of `L`.
    The initial guess is set near :math:`\frac{\pi}{4}` and uses a tight tolerance to ensure accurate results.
    The shape of the output array matches the input array `L` or `K`.
    """
    mc2 = rbamlib.constants.MC2

    # Check if K or L are not ndarray and convert them to ndarray if needed
    if not isinstance(K, np.ndarray):
        K = np.array([K])

    if not isinstance(L, np.ndarray):
        L = np.array([L])

    # Repeat K and L if they are 1D arrays
    if K.size == 1 and L.size > 1:
        K = np.repeat(K, L.size)

    # Repeat L and K if they are 1D arrays
    if L.size == 1 and K.size > 1:
        L = np.repeat(L, K.size)

    if K.size != L.size:
        raise ValueError("K and L must have the same size.")

    RHS = K / L / np.sqrt(B0(L))
    al = np.zeros_like(L)

    al_max = np.pi/2 - xtol
    al_min = 0 + xtol

    # Define the function to find roots for each element
    def equation(alpha, i):
        if alpha >= al_max:
            alpha = al_max
        if alpha <= al_min:
            alpha = al_min

        return rbamlib.models.dip.Y(alpha) / np.sin(alpha) - RHS.flat[i]

    # Solve for each element in R
    for i in range(len(np.ravel(L))):  # Flatten R to handle it as a 1D array
        tmp = root_scalar(lambda alpha: equation(alpha, i), x0=xtol, xtol=xtol)  # Initial guess near pi/4, fine tune xtol if needed
        al.flat[i] = tmp.root

    al = al.reshape(L.shape)  # Ensure al has the same shape as L

    # Convert to single value if al is a single value
    if al.size == 1:
        al = al[0]

    return al
