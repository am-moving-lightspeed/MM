import simpy
from numpy import add
from numpy import array

from task_02.qsm import QSM
from task_02.util import draw_probabilities_histograms
from task_02.util import draw_probabilities_of_amount_of_requests_in_system_histograms
from task_02.util import log_average_values_comparison
from task_02.util import log_given
from task_02.util import log_probabilities


# channels_no = 2
# max_queue_sz = 10
# requests_rt = 10
# service_rt = 5
# v_param = 1

channels_no = 5
max_queue_sz = 10
requests_rt = 100
service_rt = 10
v_param = 1

# channels_no = 2
# max_queue_sz = 1
# requests_rt = 10
# service_rt = 5
# v_param = 1

env = simpy.Environment()
model = QSM(env, channels_no, max_queue_sz, requests_rt, service_rt, v_param)

env.process(QSM.run(env, model))
env.run(until = 4000)

empiric_results = model.stats.find_empiric_probs()
theoretic_results = model.stats.find_theoretical_probs()
P_emp = empiric_results[0]
P_theor = theoretic_results[0]

values_to_compare = [values for values in zip(empiric_results[1:], theoretic_results[1:])]
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
