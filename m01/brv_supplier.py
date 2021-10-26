from typing import Tuple

from numpy import array
from numpy import ndarray
from numpy import searchsorted
from numpy import shape
from numpy import sum
from numpy.random import rand


class TwoDimensionalRandomValue:
    #
    _n_axis = 0  # vertical
    _m_axis = 1  # horizontal


    @classmethod
    def get(cls, P_matrix: ndarray, Av: ndarray, Bv: ndarray) -> Tuple[int, int]:
        n, m = shape(P_matrix)
        Qv = sum(P_matrix, axis = cls._m_axis)

        Lv = [Qv[0]]
        for i in range(1, n):
            Lv.append(Lv[i - 1] + Qv[i])

        index = searchsorted(array(Lv), rand())
        x1 = Av[index]

        Rv, k_row = [P_matrix[index][0]], P_matrix[index]
        for i in range(1, m):
            Rv.append(Rv[i - 1] + k_row[i])

        index = searchsorted(array(Rv), rand() * Rv[-1])

        return x1, Bv[index]
