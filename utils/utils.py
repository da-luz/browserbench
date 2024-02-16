import numpy as np
import pandas as pd

from scipy.stats import chi2


class BartlettSphericityChi2:
    def __init__(self, pvalue: float, statistic: float, ddof: float) -> None:
        self.pvalue = pvalue
        self.statistic = statistic
        self.ddof = ddof

    def __iter__(self):
        return iter([self.pvalue, self.statistic, self.ddof])


class PCAResult:
    def __init__(self, pvalue: float, chi2value: float, chiddof: float) -> None:
        self.pvalue = pvalue
        self.chi2value = chi2value
        self.chiddof = chiddof

    def __iter__(self):
        return iter([self.pvalue, self.chi2value, self.chiddof])


def bartletts_sphericity(df: pd.DataFrame) -> BartlettSphericityChi2:
    """Performs a Bartlett's Sphericity test for a DataFrame of numerical features
    
    Parameters
    ----------
    `df : pd.DataFrame`
        A DataFrame with n samples and k numerical features
        of wich one can extract a correlation matrix

    Returns
    -------
    `BartlettSphericityChi2`
        `pvalue : float`
        `statistic : float`
        `ddof : float`

    Raises
    ------
    `AssertionError`
        Raise this exception if the determinant of the
        correlation matrix is equals or lesser than 0 
    """


    n, k = df.shape
    cor = df.corr()

    ddof = (k * (k - 1)) / 2

    det = np.linalg.det(cor)
    assert det > 0, f"Determinant of correlation matrix = {det} <= 0"

    log = np.log(det)
    mult = (((2 * k) + 5) / 6) - (n - 1)

    statistic = round(mult * log, 5)
    pvalue = round(chi2.sf(mult * log, ddof), 5)

    return BartlettSphericityChi2(pvalue, statistic, ddof)
