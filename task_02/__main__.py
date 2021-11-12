import simpy
from matplotlib import pyplot

import qsm


Model = qsm.QSM
channels_no: int
max_queue_sz: int
requests_rt: float
service_rt: float
v_param: float

sep = '\n>>>>\n\n'

fig, axs = pyplot.subplots(2)


def log_given() -> None:
    print(sep + 'Given:')
    print(f'\tAmount of channels (n): {channels_no}')
    print(f'\tService flow rate (mu): {service_rt}')
    print(f'\tApplications flow rate (lambda): {requests_rt}')
    print(f'\tQueue waiting flow rate (v): {v_param}')
    print(f'\tMax queue length (m): {max_queue_sz}')


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


def draw_plots(model: Model) -> None:
    axs[0].hist(model.stats.time_spent_total, 50)
    axs[0].set_title('Time spent awaiting')
    axs[1].hist(model.stats.get_amounts_of_requests_in_system_at_time(), 50)

    pyplot.show()


channels_no = 2
max_queue_sz = 2
requests_rt = 3
service_rt = 4
v_param = 1

log_given()

env = simpy.Environment()
model = Model(env, channels_no, max_queue_sz, requests_rt, service_rt, v_param)

print(sep + 'Running simulation...')

env.process(Model.run(env, model))
env.run(until = 100)  # 100 minutes

empiric_values = model.stats.find_empiric_probs()
theoretic_values = model.stats.find_theoretical_probs()

print(sep + 'Empiric values:')
log_probabilities_and_average_values(*empiric_values)

print(sep + 'Theoretic values:')
log_probabilities_and_average_values(*theoretic_values)

draw_plots(model)
