import numpy as np
import datetime

def storm_idx(time, Dst, threshold=-40.0, gap_hours=1.0, method='onset'):
    r"""
    Identify storms in Dst based on a threshold, then either return their
    "onset" or "minimum Dst" index.

    Steps
    -----
    1. Find all times where Dst < threshold.
    2. Group those indices into contiguous "storm regions" separated by
       at least `gap_hours`.
    3. For each storm region:

       - If ``method='onset'``:
         a) Take the earliest index in that region (the threshold crossing),
         b) Backtrack to where Dst >= 0 or index=0.  That is the final "onset."
       - If ``method='minimum'``:
         Return the index in that region where Dst is minimum.
         Double dips remain in the same region, so only one min is reported
         per region.

    Parameters
    ----------
    time : 1D array-like of datetime.datetime
        Strictly increasing times.
    Dst : 1D array-like of float
        Dst index at the same times as `time`.
    threshold : float, default=-40.0
        Storm threshold. Values below are considered "in a storm".
    gap_hours : float, default=1.0
        If the time difference to the previous storm point is less than this,
        we treat it as the **same** storm. Otherwise, we start a new storm region.
    method : {'onset', 'minimum'}, default='onset'

        - 'onset': Return the "start index" for each storm region (after backtracking).
        - 'minimum': Return the single index at which Dst is minimal within each region.

    Returns
    -------
    storm_indices : list of int
        Indices in `time` (and `Dst`), one per identified storm, either onset or min.

    Examples
    --------
    >>> import datetime
    >>> from rbamlib.web import omni
    >>> time, Dst = omni('20131001', '20131101', {'Dst'})
    >>> # ONSET method
    >>> s_onsets = storms_idx(time, Dst, threshold=-40, gap_hours=1.0, method='onset')
    >>> # MINIMUM method
    >>> s_mins = storms_idx(time, Dst, threshold=-40, gap_hours=1.0, method='minimum')
    """
    # Ensure arrays
    time = np.asarray(time)
    Dst  = np.asarray(Dst)

    if len(time) != len(Dst):
        raise ValueError("time and Dst must have the same length.")

    # 1) Find indices where Dst < threshold
    below = np.where(Dst < threshold)[0]
    if not len(below):
        return []

    # 2) Build contiguous "storm regions" separated by >= gap_hours
    #    i.e., if the current index is within gap_hours of the previous, we group them.
    #    We'll store them as a list of lists.
    storm_regions = []
    current_region = [below[0]]

    def hours_diff(t1, t2):
        """Return (t1 - t2) in hours, where t1, t2 are datetimes."""
        return (t1 - t2).total_seconds() / 3600.0

    for idx in below[1:]:
        dt_hrs = hours_diff(time[idx], time[current_region[-1]])
        if dt_hrs < gap_hours:
            # same storm region
            current_region.append(idx)
        else:
            # new storm region
            storm_regions.append(current_region)
            current_region = [idx]

    # Add the final region
    storm_regions.append(current_region)

    # 3) For each storm region, choose an index based on `method`
    results = []
    for region in storm_regions:
        if method == 'onset':
            # Take the earliest crossing
            onset_idx = region[0]
            # Then "backtrack" until Dst[onset_idx] >= 0 or onset_idx=0
            while onset_idx > 0 and Dst[onset_idx] < 0:
                onset_idx -= 1
            results.append(onset_idx)

        elif method == 'minimum':
            # Find the index in region for which Dst is minimal
            region_dsts = Dst[region]
            min_idx_local = np.argmin(region_dsts)  # index within `region`
            # Convert to global index
            min_idx_global = region[min_idx_local]
            results.append(min_idx_global)

        else:
            raise ValueError("method must be 'onset' or 'minimum'")

    # Sort + unique in case of edge merges or overlaps (unlikely but safe)
    results = np.unique(results)
    return results
