# Anderson–Darling statistic implementation stub

import numpy as np

def ad_statistic(sample, cdf):
    """
    Compute the Anderson–Darling (AD) statistic for a 1D sample and a given CDF function.

    Parameters:
        sample (array-like): 1D array or list of sample data.
        cdf (callable): Function to compute the CDF of the hypothesized distribution.

    Returns:
        float: The AD statistic.
    """
    sample = np.asarray(sample)
    if sample.ndim != 1:
        raise ValueError("Sample must be 1-dimensional.")
    n = len(sample)
    if n == 0:
        raise ValueError("Sample must not be empty.")
    sorted_sample = np.sort(sample)
    cdf_vals = cdf(sorted_sample)
    # Avoid log(0) by clipping
    cdf_vals = np.clip(cdf_vals, 1e-10, 1-1e-10)
    i = np.arange(1, n+1)
    S = np.sum((2*i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
    ad = -n - S/n
    return ad
