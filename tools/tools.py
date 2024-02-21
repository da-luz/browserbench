import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from scipy.stats import chi2, pearsonr


class BartlettSphericityChi2:
    def __init__(self, pvalue: float, statistic: float, ddof: float) -> None:
        self.pvalue = pvalue
        self.statistic = statistic
        self.ddof = ddof

    def __iter__(self):
        return iter([self.pvalue, self.statistic, self.ddof])

    def __str__(self) -> str:
        stat = "{:.5f}".format(self.statistic)
        pval = "{:.5f}".format(self.pvalue)
        dof = "{:.1f}".format(self.ddof)

        test = "Bartlett's chi squared test for sphericity\n"
        test += "-" * (len(test) - 1)
        test += f"\nchi2: {stat}, ddof: {dof}"
        test += f"\npvalue: {pval}"
        return test


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


class PCA:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.corr = df.corr()

    def fit(self, n_factors: int | str = 'auto', print_eigenvalues: bool = True):
        self.n_factors = n_factors

        self.linear_combination(print_eigenvalues)
        self.factor_loadings()

    def linear_combination(self, print_eigenvalues: bool = True):
        eigen = np.linalg.eig(self.corr)
        values = eigen.eigenvalues
        vectors = eigen.eigenvectors

        if print_eigenvalues: print(f"Eigenvalues: {sorted(values)[::-1]}")
        if type(self.n_factors) == int:
            print(f"Extracting {self.n_factors} factors")
            extract = [list(values).index(v) for v in sorted(values)[::-1][:self.n_factors]]
        else:
            match self.n_factors:
                case 'auto':
                    extract = [list(values).index(v) for v in sorted(values)[::-1] if v > 1]
                    print(f"Extracting {len(extract)} factors based on Kaiser's criterion")
                case 'all':
                    print("Extracting all factors from dataset")
                    extract = [list(values).index(v) for v in sorted(values)[::-1]]

        self.eigenvalues = values[extract]
        self.eigenvectors = vectors[:, extract]

    def factor_loadings(self):
        self.score = self.eigenvectors / np.sqrt(self.eigenvalues)
        self.factors = scale(self.df) @ self.score
        factor_load = []

        for feature in self.df.columns:
            xl = []
            x = self.df[feature].values
            for y in self.factors.T:
                xl.append(pearsonr(x, y).statistic)
            factor_load.append(xl)

        self.factors = pd.DataFrame(self.factors, index=self.df.index)
        self.factor_load = pd.DataFrame(factor_load, index=self.df.columns)
        self.communalities = pd.DataFrame((self.factor_load ** 2).sum(1), columns=['Communalities'])
