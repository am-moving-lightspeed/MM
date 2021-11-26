from matplotlib import pyplot as plt
from numpy import append
from numpy import array
from numpy import array_split
from numpy import ndarray


def log_probabilities_and_average_values(*args) -> None:
    (P, A, P_rejected,
     avg_req_in_system, avg_req_in_queue,
     avg_req_time_in_system, avg_req_time_in_queue,
     avg_busy_channels) = args

    print('\tP-probabilities:')
    for i, prob in enumerate(P):
        print(f'\t\tP{i}: {prob}')
    print(f'\tAbsolute throughput: {A}')
    print(f'\tProbability for a request to be rejected: {P_rejected}')
    print(f'\tAverage amount of requests in the system: {avg_req_in_system}')
    print(f'\tAverage amount of requests in a queue: {avg_req_in_queue}')
    print(f'\tAverage time of request spent in the system: {avg_req_time_in_system}')
    print(f'\tAverage time of request spent in the queue: {avg_req_time_in_queue}')
    print(f'\tAverage amount of busy channels: {avg_busy_channels}')


def draw_P_comparison_histograms(*args, titles) -> None:
    amount = len(args)
    plt.style.use('default')
    fig, axs = plt.subplots(1, amount, figsize = (25, 5))

    for ind, P_emp in enumerate(args):
        axs[ind].bar(range(len(P_emp)), P_emp, width = 0.5)
        axs[ind].set_title(titles[ind])

    plt.show()


def draw_P_sub_comparison_histograms(*args, titles) -> None:
    amount = len(args)
    plt.style.use('default')
    fig, axs = plt.subplots(1, amount, figsize = (25, 5))

    for ind, (P_emp, P_theor) in enumerate(args):
        P_emp, P_theor = array(P_emp), array(P_theor)
        axs[ind].bar(range(len(P_emp)), P_emp - P_theor, width = 0.5)
        axs[ind].axhline(y = 0, xmin = 0, xmax = len(P_emp), color = 'red')
        axs[ind].set_title(titles[ind])

    plt.show()


def draw_values_comparison_bars(*args, titles):
    amount = len(args)
    plt.style.use('default')
    fig, axs = plt.subplots(1, amount, figsize = (25, 5))

    for ind, actual_expected_pair in enumerate(args):
        axs[ind].bar((1, 2), actual_expected_pair, width = 0.2)
        axs[ind].axhline(y = actual_expected_pair[1], xmin = 0, xmax = 2, color = 'red')
        axs[ind].set_title(titles[ind])

    plt.show()


def draw_probabilities_of_amount_of_requests_in_system_histograms(*args, count):
    amount = len(args)
    plt.style.use('default')
    fig, axs = plt.subplots(1, amount, figsize = (15, 5))

    for ind, (requests_in_system_at_time, theoretical_probs) in enumerate(args):
        _draw_probabilities_of_amount_of_requests_in_system_histograms(
          axs[ind]
          , count
          , requests_in_system_at_time
          , theoretical_probs
        )

    plt.show()


def _draw_probabilities_of_amount_of_requests_in_system_histograms(
  axs
  , index
  , requests_in_system_at_time: ndarray
  , theoretical_probs: ndarray):
    #
    interval_len = 100
    intervals = array_split(requests_in_system_at_time, interval_len)

    for i in range(1, len(intervals)):
        intervals[i] = append(intervals[i], intervals[i - 1])

    interval_probabilities = []
    for interval in intervals:
        interval_probabilities.append(len(interval[interval == index]) / len(interval))

    axs.bar(range(len(interval_probabilities)), interval_probabilities)
    axs.set_title(f"Prob. that there is ({index}) req. in system at time")
    axs.axhline(y = theoretical_probs[index], xmin = 0, xmax = len(interval_probabilities),
                color = 'red')
