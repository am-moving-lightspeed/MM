import math
from collections import Counter
from typing import Callable
from typing import List
from typing import Union

import scipy.stats.mstats as mstats
from numpy import ndarray
from numpy import shape
from numpy import sum
from numpy import zeros
from scipy.stats import chi2
from scipy.stats import t


class Investigator:

    def __init__(self, P_matrix: ndarray, Av: ndarray, Bv: ndarray, supplier: Callable):
        self._P_matrix = P_matrix
        self._Av = Av
        self._Bv = Bv
        self._supplier = supplier


    def build_empiric_matrix(self, sample_size: int) -> ndarray:
        d_matrix = zeros(shape(self._P_matrix))
        sample = Counter(
          [self._supplier(self._P_matrix, self._Av, self._Bv) for _ in range(sample_size)]
        )

        Av, Bv = self._Av.tolist(), self._Bv.tolist()
        for (x1, x2), count in sample.items():
            i = Av.index(x1)
            j = Bv.index(x2)
            d_matrix[i][j] = count / sample_size

        return d_matrix


    @staticmethod
    def M_point_estimation(X, n) -> float:
        return math.fsum(X) / n


    @staticmethod
    def D_point_estimation(X, n, M_estimate) -> float:
        return 1 / (n - 1) * math.fsum(list(map(lambda xi: (xi - M_estimate) ** 2, X)))


    @staticmethod
    def M_confidence_interval(n, D, M, probability) -> List[Union[float, List[float]]]:
        arr = t(n).rvs(100000)

        delta = mstats.mquantiles(arr, prob = probability) * math.sqrt(D / (n - 1))
        confidence_interval = [M - delta[0], M + delta[0]]

        return [delta, confidence_interval]


    @staticmethod
    def D_confidence_interval(n, D, M, probabilities) -> List[Union[float, List[float]]]:
        arr = chi2(n - 1).rvs(100000)

        delta = mstats.mquantiles(arr, prob = probabilities)
        confidence_interval = [D - (n - 1) * D / delta[1], D + (n - 1) * D / delta[0]]

        return [delta[1], confidence_interval]


    @staticmethod
    def find_correlation(M_x1, M_x2, D_x1, D_x2, empiric_matrix, A, B):
        M_x1_x2 = 0
        for i in range(empiric_matrix.shape[0]):
            for j in range(empiric_matrix.shape[1]):
                M_x1_x2 += A[i] * B[j] * empiric_matrix[i, j]
        return (M_x1_x2 - M_x1 * M_x2) / math.sqrt(D_x1) * math.sqrt(D_x2)


    @staticmethod
    def pearson_criterion(theoretical_matrix, empiric_matrix, n):
        chi2_ = n * sum((empiric_matrix - theoretical_matrix) ** 2 / theoretical_matrix)
        chi2_value = chi2.ppf(0.95, theoretical_matrix.size - 1)
        return chi2_ < chi2_value
