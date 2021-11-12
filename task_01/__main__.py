import matplotlib.pyplot as plt
from numpy import array
from numpy import ndarray
from numpy import sum

import task_01.brv_supplier as sup
import task_01.investigator as inv


fig, axs = plt.subplots(3, 2)


def draw_histogram(A, B, empiric_matrix):
    x1_probability = sum(empiric_matrix, axis = 1)
    x2_probability = sum(empiric_matrix, axis = 0)
    axs[0][0].bar(A, x1_probability)
    axs[0][1].bar(B, x2_probability, color = 'red')
    axs[0][0].legend(['X1'])
    axs[0][1].legend(['X2'])


def print_interval_estimation(no: int, n, M, D, probabilities_for_M_int,
                              probabilities_for_D_int):
    print(f'<<<----     Интервальные оценки для x{no}     ---->>>', '\nДля мат. ожидания:')
    deltas = []
    for prob in probabilities_for_M_int:
        delta, interval = investigator.M_confidence_interval(n, M, D, prob)
        deltas.append(delta)
        print('Доверительный интервал для мат. ожидания при доверительной вероятности',
              f'{prob} и n = {n}: {str(interval)}')
    axs[1][no - 1].plot(probabilities_for_M_int, list(map(lambda x: x * 2, deltas)))

    print('\nДля дисперсии:')
    deltas = []
    D_probs = []
    intervals = []
    for prob in probabilities_for_D_int:
        delta, interval = investigator.D_confidence_interval(n, D, M, prob)
        deltas.append(delta)
        D_probs.append(prob[1])
        intervals.append(interval)
        print('Доверительный интервал для дисперсии при доверительной вероятности',
              f'{prob[1]} и n = {n}: {str(interval)}')
    axs[2][no - 1].plot(D_probs, list(map(lambda x: x[1] - x[0], intervals)))
    print('>>>----     --------------------------     ----<<<\n')


P = array([
  [0.2, 0.3],
  [0.1, 0.2],
  [0.1, 0.1]
])
A = array([1, 2, 4])
B = array([1, 3])
print('Теоретическая матрица:')
print(P, "\n")

empiric_matrix: ndarray
investigator = inv.Investigator(P, A, B, sup.TwoDimensionalRandomValue.get)
print('Эмпирическая матрица (n = 100):\n', investigator.build_empiric_matrix(100), '\n')
print('Эмпирическая матрица (n = 1000):\n', investigator.build_empiric_matrix(1000), '\n')
print('Эмпирическая матрица (n = 10000):\n', investigator.build_empiric_matrix(10000), '\n')
print('Эмпирическая матрица (n = 100000):\n',
      empiric_matrix := investigator.build_empiric_matrix(100000),
      '\n')
draw_histogram(A, B, empiric_matrix)

n = 10000
X = array([sup.TwoDimensionalRandomValue.get(P, A, B) for i in range(n)])
x1v, x2v = X[:, 0], X[:, 1]

M_x1 = investigator.M_point_estimation(x1v, n)
M_x2 = investigator.M_point_estimation(x2v, n)
print('Точечные оценки мат. ожидания:\n\tДля x1:', str(M_x1), '\n\tДля x2:', str(M_x2), '\n')

D_x1 = investigator.D_point_estimation(x1v, n, M_x1)
D_x2 = investigator.D_point_estimation(x2v, n, M_x2)
print('Точечные оценки дисперсии:\n\tДля x1:', str(D_x1), '\n\tДля x2:', str(D_x2), '\n')

probabilities_for_M_int = [0.9, 0.95, 0.98, 0.99]
probabilities_for_D_int = [[0.025, 0.975], [0.01, 0.99], [0.005, 0.995]]
print_interval_estimation(1, n, M_x1, D_x1, probabilities_for_M_int, probabilities_for_D_int)
print_interval_estimation(2, n, M_x2, D_x2, probabilities_for_M_int, probabilities_for_D_int)

print('Коэффициент корреляции:')
print(investigator.find_correlation(M_x1, M_x1, D_x1, D_x2, empiric_matrix, A, B))
if investigator.pearson_criterion(P, empiric_matrix, n):
    print('Фактические данные не противоречат ожидаемым по критерию Пирсона.')

plt.show()
