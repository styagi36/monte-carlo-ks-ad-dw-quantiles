# Durbin–Watson statistic implementation stub

import numpy as np

def dw_statistic(residuals):
    """
    Compute the Durbin–Watson (DW) statistic for regression residuals.

    Parameters:
        residuals (array-like): 1D array or list of regression residuals.

    Returns:
        float: The DW statistic.
    """
    residuals = np.asarray(residuals)
    if residuals.ndim != 1:
        raise ValueError("Residuals must be 1-dimensional.")
    n = len(residuals)
    if n < 2:
        raise ValueError("Need at least 2 residuals for DW statistic.")
    diff = np.diff(residuals)
    dw = np.sum(diff**2) / np.sum(residuals**2)
    return dw
