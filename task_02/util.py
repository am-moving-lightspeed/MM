import matplotlib.pyplot as plt
import prettytable as tbl
from numpy import append
from numpy import around
from numpy import array_split
from numpy import ndarray


sep = '\n>>>>\n'
_round_to = 10


def log_given(*args) -> None:
    (channels_no, max_queue_sz, requests_rt, service_rt, v_param) = args

    print(sep)
    print('Given:')
    print(f'\tAmount of channels (n): {channels_no}')
    print(f'\tService flow rate (mu): {service_rt}')
    print(f'\tApplications flow rate (lambda): {requests_rt}')
    print(f'\tQueue waiting flow rate (v): {v_param}')
    print(f'\tMax queue length (m): {max_queue_sz}')


def log_probabilities(P_emp: ndarray, P_theor: ndarray) -> None:
    final_probabilities_info = tbl.PrettyTable()

    final_probabilities_info.add_column("Эмпирические финальные вероятности",
                                        around(P_emp, _round_to))
    final_probabilities_info.add_column("Теоретические финальные вероятности",
                                        around(P_theor, _round_to))
    print(sep)
    print(final_probabilities_info)


def log_average_values_comparison(*args) -> None:
    field_names = [
      "Относитальная пропускная способность"
      , "Абсолютная пропускная способность"
      , "Вероятность отказа"
      , "Среднее число элементов в СМО"
      , "Среднее число элементов в очереди"
      , "Среднее время пребывания элемента в СМО"
      , "Среднее время пребывания элемента в очереди"
    ]

    print(sep)
    for i, value in enumerate(field_names):
        info = tbl.PrettyTable()
        info.field_names = ["Исследование", value]
        info.add_row(["Теоретическое", around(args[i][0], _round_to)])
        info.add_row(["Эмпирическое", around(args[i][1], _round_to)])
        print(info)


def draw_probabilities_histograms(P_emp: ndarray, P_theor: ndarray) -> None:
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (25, 5))

    ax1.set_title('Эмпирические финальные вероятности')
    ax1.bar(range(len(P_emp)), P_emp, width = 0.5)

    ax2.set_title('Теоретические финальные вероятности')
    ax2.bar(range(len(P_theor)), P_theor, width = 0.5)

    ax3.set_title('Финальные вероятности')
    ax3.bar(range(len(P_theor)), P_theor - P_emp, width = 0.5)
    ax3.axhline(y = 0, xmin = 0, xmax = len(P_theor), color = 'red')

    plt.show()


def draw_probabilities_of_amount_of_requests_in_system_histograms(
  requests_in_system_at_time: ndarray,
  theoretical_probs: ndarray):
    #
    interval_len = 100
    intervals = array_split(requests_in_system_at_time, interval_len)

    for i in range(1, len(intervals)):
        intervals[i] = append(intervals[i], intervals[i - 1])

    for i in range(len(theoretical_probs)):
        interval_probabilities = []
        for interval in intervals:
            interval_probabilities.append(len(interval[interval == i]) / len(interval))
        plt.figure(figsize = (5, 5))
        plt.bar(range(len(interval_probabilities)), interval_probabilities)
        plt.title(f"Probability that there is ({i}) requests in system at time")
        plt.axhline(y = theoretical_probs[i], xmin = 0, xmax = len(interval_probabilities),
                    color = 'red')
        plt.show()
