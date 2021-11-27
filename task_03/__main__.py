from math import factorial
from math import inf
from typing import Tuple

import simpy
from numpy import add
from numpy import array
from numpy import prod

from task_02.qsm import QSM
from task_02.util import log_average_values_comparison
from task_02.util import log_probabilities
from task_03.util import draw_P_comparison_histograms
from task_03.util import draw_P_sub_comparison_histograms
from task_03.util import draw_probabilities_of_amount_of_requests_in_system_histograms
from task_03.util import draw_values_comparison_bars


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
    avg_req_processed = sts.get_average_amount_of_requests_processed_at_time()
    avg_req_in_queue = 0
    avg_req_time_in_system = sts.get_average_time_of_request_spent_in_system()
    avg_req_time_in_queue = 0

    return (P, Q, A, P_rejected, avg_req_in_system, avg_req_in_queue, avg_req_time_in_system,
            avg_req_time_in_queue, avg_req_processed)


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
    avg_req_processed = A / model.service_rt
    avg_req_in_queue = 0
    avg_req_time_in_system = 1 / model.service_rt
    avg_req_time_in_queue = 0

    return (P, Q, A, P_rejected, avg_req_in_system, avg_req_in_queue, avg_req_time_in_system,
            avg_req_time_in_queue, avg_req_processed)


####

# channels_no = inf
# max_queue_sz = inf
# requests_rt = 7  # ~ 7 requests arrive per minute
# service_rt = 0.2  # ~ 1 request processed per 5 minutes
# v_param = 1  # not expected to participate

channels_no = inf
max_queue_sz = inf
service_rt = 0.4
v_param = 1.5

#### case 1

requests_rt_1 = 15

env = simpy.Environment()
model = QSM(env, channels_no, max_queue_sz, requests_rt_1, service_rt, v_param)
env.process(QSM.run(env, model))
env.run(until = 2000)

empiric_values_1 = find_empiric_probs(model)
P_dim = len(empiric_values_1[0]) - 1
theoretic_values_1 = find_theoretical_probs(model, P_dim, 1)

values_to_compare_1 = [values for values in zip(empiric_values_1[1:], theoretic_values_1[1:])]
requests_in_system_at_time_1 = add(
  array(model.stats.requests_processing)
  , array(model.stats.requests_awaiting)
)

P_emp_1 = empiric_values_1[0]
P_theor_1 = theoretic_values_1[0]

log_probabilities(P_emp_1, P_theor_1)
log_average_values_comparison(*values_to_compare_1)

#### case 2

requests_rt_2 = 3

env = simpy.Environment()
model = QSM(env, channels_no, max_queue_sz, requests_rt_2, service_rt, v_param)
env.process(QSM.run(env, model))
env.run(until = 2000)

empiric_values_2 = find_empiric_probs(model)
P_dim = len(empiric_values_2[0]) - 1
theoretic_values_2 = find_theoretical_probs(model, P_dim, 1)

values_to_compare_2 = [values for values in zip(empiric_values_2[1:], theoretic_values_2[1:])]
requests_in_system_at_time_2 = add(
  array(model.stats.requests_processing)
  , array(model.stats.requests_awaiting)
)

P_emp_2 = empiric_values_2[0]
P_theor_2 = theoretic_values_2[0]

log_probabilities(P_emp_2, P_theor_2)
log_average_values_comparison(*values_to_compare_2)

#### case 3

requests_rt_3 = 32

env = simpy.Environment()
model = QSM(env, channels_no, max_queue_sz, requests_rt_3, service_rt, v_param)
env.process(QSM.run(env, model))
env.run(until = 2000)

empiric_values_3 = find_empiric_probs(model)
P_dim = len(empiric_values_3[0]) - 1
theoretic_values_3 = find_theoretical_probs(model, P_dim, 1)

values_to_compare_3 = [values for values in zip(empiric_values_3[1:], theoretic_values_3[1:])]
requests_in_system_at_time_3 = add(
  array(model.stats.requests_processing)
  , array(model.stats.requests_awaiting)
)

P_emp_3 = empiric_values_3[0]
P_theor_3 = theoretic_values_3[0]

log_probabilities(P_emp_3, P_theor_3)
log_average_values_comparison(*values_to_compare_3)

####

prefix = 'When requests rate is %d'
titles = [prefix % requests_rt_1, prefix % requests_rt_2, prefix % requests_rt_3]
draw_P_comparison_histograms(P_emp_1, P_emp_2, P_emp_3, titles = titles)
draw_P_comparison_histograms(P_theor_1, P_theor_2, P_theor_3, titles = titles)
draw_P_sub_comparison_histograms(
  (P_emp_1, P_theor_1)
  , (P_emp_2, P_theor_2)
  , (P_emp_3, P_theor_3)
  , titles = titles
)

draw_values_comparison_bars(
  values_to_compare_1[3]
  , values_to_compare_2[3]
  , values_to_compare_3[3]
  , titles = ['Сред. число заявок в СМО'] * 3
)
draw_values_comparison_bars(
  values_to_compare_1[5]
  , values_to_compare_2[5]
  , values_to_compare_3[5]
  , titles = ['Сред. время пребывания заявки в СМО'] * 3
)

for i in range(min(len(P_theor_1), len(P_theor_2), len(P_theor_3))):
    draw_probabilities_of_amount_of_requests_in_system_histograms(
      (requests_in_system_at_time_1, P_theor_1)
      , (requests_in_system_at_time_2, P_theor_2)
      , (requests_in_system_at_time_3, P_theor_3)
      , count = i
    )
