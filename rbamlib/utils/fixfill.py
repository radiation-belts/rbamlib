import numpy as np
from scipy.interpolate import interp1d

def fixfill(time, data, fillval, method='nan', fillval_mode='eq'):
    """
    Fix invalid values in time-series data using NaN replacement or interpolation.

    This function identifies missing or invalid data based on a `fillval` rule,
    marks those values as NaN, and optionally interpolates over them using a
    linear scheme.

    Parameters
    ----------
    time : ndarray of datetime.datetime
        1D array of strictly increasing datetime objects.
    data : ndarray
        1D array of numerical values corresponding to `time`.
    fillval : float
        The fill value or threshold to identify missing data.
    method : {'nan', 'interp'}, default='nan'

        - 'nan': Replace missing values with NaN.
        - 'interp': Replace and linearly interpolate missing values.
    fillval_mode : {'eq', 'gt', 'lt'}, default='equal'

        - 'eq':    treat `fillval` as an exact match, equal.
        - 'gt':    treat all `data >= fillval` as missing.
        - 'lt':    treat all `data <= fillval` as missing.

    Returns
    -------
    fixed_data : ndarray
        The cleaned data array with missing values replaced or interpolated.

    Examples
    --------
    >>> time = [datetime(2023,1,1,0,0) + timedelta(minutes=5*i) for i in range(6)]
    >>> data = np.array([1.0, 2.0, 999, 4.0, 999, 6.0])

    Mark fill values as NaN:
    >>> fixfill(time, data, fillval=999, method='nan', fillval_mode='equal')
    array([ 1.,  2., nan,  4., nan,  6.])

    Replace fill values and interpolate linearly:
    >>> fixfill(time, data, fillval=999, method='interp', fillval_mode='equal')
    array([1., 2., 3., 4., 5., 6.])
    """

    data = np.array(data, dtype=float)

    # Apply fillval condition
    if fillval_mode == 'eq':
        data[data == fillval] = np.nan
    elif fillval_mode == 'gt':
        data[data >= fillval] = np.nan
    elif fillval_mode == 'lt':
        data[data <= fillval] = np.nan
    else:
        raise ValueError("fillval_mode must be 'equal', 'gt', or 'lt'.")

    if method == 'nan':
        return data

    elif method == 'interp':
        mask_valid = ~np.isnan(data)
        if mask_valid.sum() < 2:
            return data

        # Convert datetimes to elapsed seconds
        time_seconds = np.array([(t - time[0]).total_seconds() for t in time])

        interp_func = interp1d(
            time_seconds[mask_valid], data[mask_valid],
            kind='linear', bounds_error=False, fill_value=np.nan
        )

        return interp_func(time_seconds)

    else:
        raise ValueError("method must be 'nan' or 'interp'.")
