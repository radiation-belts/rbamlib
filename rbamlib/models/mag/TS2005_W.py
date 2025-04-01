import numpy as np


def TS2005_W(time, S, storm_onsets=None, fill_value=np.nan):
    r"""
    Compute Tsyganenko & Sitnov (2005) [#]_ :math:`W_k(t_i)`, per Eq. (7) with optional "storm-by-storm" resets.

    Parameters
    ----------
    time : 1D array-like of datetime.datetime
        Strictly increasing times.  Must match the length of the second dimension of S.
    S : 2D ndarray, shape (N, 6)
        The source function :math:`S_k(t_i)` (from eq. (8)).
        6 columns for k=1..6, N rows for times 0..N-1.
    storm_onsets : list or 1D array of int, default=None
        Indices of `time` that mark the beginning of each storm.
        If provided, then for i in [onsets[m]..onsets[m+1]-1], we sum from onsets[m].
        Times before the first onset or after the last are filled with `fill_value`.
        If None (default), we sum from i=0 (the earliest time). Add 0 to storm_onsets
        array to include the calculation of coefficients before the storm.
    fill_value : float, default=np.nan
        Value for W outside any identified storms. Default is `NaN`.

    Returns
    -------
    W : 2D ndarray, shape (N, 6)
        6 different :math:`W_k(t_i)` for each k (and i).

    Notes
    -----

    Equation (7) in Tsyganenko & Sitnov (2005) can be written (for each k):

    .. math::
        W_k(t_i) = \frac{r_k}{12} \sum_{j = j_{\mathrm{start}}}^{i}
            \bigl[ S_k(t_j)\,\exp\bigl(r_k \times (t_j - t_i)\times 24 \bigr)\bigr],

    Here:
      - :math:`(t_j - t_i)` is in **days**,
      - :math:`r_k` is provided in :math:`\mathrm{hr}^{-1}`,
      - The factor :math:`\times 24` converts days to hours.
      - :math:`j_{\mathrm{start}}` = 0 if summing from the very first sample, or
        the index of the most recent storm onset (if `storm_onsets` is provided).

    References
    ----------
    .. [#] Tsyganenko, N. A., & Sitnov, M. I. (2005). Modeling the dynamics of the inner magnetosphere during strong geomagnetic storms. Journal of Geophysical Research, 110(A3), 7737. https://doi.org/10.1029/2004JA010798
    """
    # Relaxation rates (hours^-1) for the 6 T05 coefficients
    r_vals = np.array([0.39, 0.70, 0.031, 0.58, 1.15, 0.88])
    n_times, n_coeff = S.shape
    if n_coeff != 6:
        raise ValueError(f"S must have shape (N, 6), not {S.shape}.")

    # Convert time to array of datetime
    time = np.asarray(time)

    # Prepare output array
    W = np.full((n_times, 6), fill_value, dtype=float)

    # If no storm onsets => single segment from 0..n_times
    if storm_onsets is None:
        storm_onsets = np.array([0, n_times])
    else:
        # Ensure sorted, unique
        storm_onsets = np.unique(storm_onsets)
        # Filter any that are out-of-bounds
        storm_onsets = storm_onsets[(storm_onsets >= 0) & (storm_onsets < n_times)]
        # Append a sentinel at the end if needed
        if storm_onsets[-1] < n_times:
            storm_onsets = np.append(storm_onsets, n_times)

    # Process each storm segment separately
    for idx_start, idx_end in zip(storm_onsets, storm_onsets[1:]):
        if idx_start >= idx_end:
            continue

        # Extract the segment and convert times to numeric days relative to segment start
        seg_times = time[idx_start:idx_end]
        t_seg = np.array([(t - seg_times[0]).total_seconds() / 86400.0 for t in seg_times])
        S_seg = S[idx_start:idx_end, :]  # shape (n_seg, 6)
        n_seg = len(t_seg)

        # # Sum using Matrix
        # # Precompute the lower-triangular mask for the segment
        # L = np.tril(np.ones((n_seg, n_seg)))
        #
        # # For each coefficient, vectorize the inner summation
        # for k in range(6):
        #     # dt matrix: dt[i, j] = t_seg[j] - t_seg[i]
        #     dt = t_seg[None, :] - t_seg[:, None]  # shape (n_seg, n_seg)
        #     # Apply L mask, so we do not compute large exp (inf)
        #     dt = dt * L
        #     # Only consider terms where j <= i by applying the mask L
        #     kernel = np.exp(r_vals[k] * dt * 24) * L  # shape (n_seg, n_seg)
        #     # For each i, sum over j from 0 to i:
        #     #    sum_{j=0}^{i} S_seg[j, k] * exp( r_k * (t_seg[j]-t_seg[i])*24 )
        #     sum_vals = np.sum(kernel * S_seg[:, k][None, :], axis=1)
        #     # Multiply by r_k/12 to get W_seg for this coefficient
        #     W_seg = (r_vals[k] / 12.0) * sum_vals
        #     # Store the computed W values back into the output array
        #     W[idx_start:idx_end, k] = W_seg

        # # Sum using loop
        # Preallocate W_seg for the current segment
        W_seg = np.zeros(n_seg)
        for k in range(6):
            sum_val = 0.
            for i in range(n_seg):
                # Compute differences only for j=0 to i
                dt_row = t_seg[:i + 1] - t_seg[i]
                # Compute the kernel for this row
                kernel_val = np.exp(r_vals[k] * dt_row * 24)
                # Sum only over the valid indices (j <= i)
                sum_val = np.sum(S_seg[:i + 1, k] * kernel_val)
                W_seg[i] = (r_vals[k] / 12.0) * sum_val
            # Store computed values into output array
            W[idx_start:idx_end, k] = W_seg

    return W
