from math import factorial
from math import inf
from typing import Tuple

import simpy
from numpy import add
from numpy import array
from numpy import prod

from task_02.qsm import QSM
from task_02.util import draw_probabilities_histograms
from task_02.util import draw_probabilities_of_amount_of_requests_in_system_histograms
from task_02.util import log_average_values_comparison
from task_02.util import log_given
from task_02.util import log_probabilities


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


def find_empiric_probs(model: QSM) -> Tuple:
    sts = model.stats

    """
    Since there are infinite amount of channels, no rejections are possible
    """
    requests_passed_total = sts.requests_completed_amount

    """
    P -- array of probabilities. P[i] is the probability, that
    req_proc_and_await_at_time[i] < i.
    """
    P = []
    """
    req_proc_and_await_at_time -- vector, where each component shows how many
    requests there had been processing and in a queue by the time a new request came.
    """
    req_proc_and_await_at_time = array(sts.requests_processing)
    max_req_proc_and_await_amount = len(req_proc_and_await_at_time)
    for i in range(1, max_req_proc_and_await_amount + 1):
        matches = req_proc_and_await_at_time[req_proc_and_await_at_time == i]
        if prob := (len(matches) / requests_passed_total):
            P.append(prob)

    """
    P_rejected -- probability for a request to be rejected.
    """
    P_rejected = 0

    """
    Q -- probability for a request to be processed.
    """
    Q = 1 - P_rejected

    """
    A -- absolute throughput.
    """
    A = Q * sts.model.requests_rt

    avg_busy_channels = A / sts.model.service_rt

    avg_req_in_system = sts.get_average_amount_of_requests_in_system_at_time()
    avg_req_in_queue = 0
    avg_req_time_in_system = sts.get_average_time_of_request_spent_in_system()
    avg_req_time_in_queue = 0

    return (P, Q, A, P_rejected, avg_req_in_system, avg_req_in_queue, avg_req_time_in_system,
            avg_req_time_in_queue)


def find_theoretical_probs(model: QSM, channels_no: int, max_queue_sz: int) -> Tuple:
    sts = model.stats

    ro = model.requests_rt / model.service_rt
    betta = model.v_param / model.service_rt

    P: list = []
    p00 = sum([ro ** i / factorial(i) for i in range(channels_no + 1)])
    p01 = (ro ** channels_no / factorial(channels_no))
    p02 = sum([
      ro ** i / (prod([channels_no + t * betta for t in range(1, i + 1)]))
      for i in range(1, max_queue_sz + 1)
    ])
    p0 = (p00 + p01 * p02) ** -1

    P.append(p0)
    for i in range(1, channels_no + 1):
        px = (ro ** i / factorial(i)) * p0
        P.append(px)

    P_rejected = 0
    Q = 1
    A = Q * model.requests_rt
    avg_req_in_system = model.requests_rt * 1 / model.service_rt
    avg_req_in_queue = 0
    avg_req_time_in_system = 1 / model.service_rt
    avg_req_time_in_queue = 0

    return (P, Q, A, P_rejected, avg_req_in_system, avg_req_in_queue, avg_req_time_in_system,
            avg_req_time_in_queue)

####

# channels_no = inf
# max_queue_sz = inf
# requests_rt = 7  # ~ 7 requests arrive per minute
# service_rt = 0.2  # ~ 1 request processed per 5 minutes
# v_param = 1  # not expected to participate

channels_no = inf
max_queue_sz = inf
requests_rt = 15
service_rt = 0.4
v_param = 1.5

####

env = simpy.Environment()
model = QSM(env, channels_no, max_queue_sz, requests_rt, service_rt, v_param)

env.process(QSM.run(env, model))
env.run(until = 100)  # 300 minutes

empiric_values = find_empiric_probs(model)
P_dim = len(empiric_values[0]) - 1
theoretic_values = find_theoretical_probs(model, P_dim, 1)

P_emp = empiric_values[0]
P_theor = theoretic_values[0]

values_to_compare = [values for values in zip(empiric_values[1:], theoretic_values[1:])]
requests_in_system_at_time = add(
  array(model.stats.requests_processing)
  , array(model.stats.requests_awaiting)
)

####

log_given(channels_no, max_queue_sz, requests_rt, service_rt, v_param)

log_probabilities(P_emp, P_theor)

log_average_values_comparison(*values_to_compare)

draw_probabilities_histograms(array(P_emp), array(P_theor))

draw_probabilities_of_amount_of_requests_in_system_histograms(requests_in_system_at_time,
                                                              P_theor)
