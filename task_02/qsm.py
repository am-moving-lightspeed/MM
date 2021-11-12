import math

from numpy.random import exponential
from simpy import Environment
from simpy.resources.resource import Request
from simpy.resources.resource import Resource

from task_02.statistics import Statistics


class QSM:

    def __init__(self
                 , env: Environment
                 , channels_no: int
                 , max_queue_sz: int
                 , requests_rt: float
                 , service_rt: float
                 , v_param: float):
        self.env = env
        self.channels_no = channels_no
        self.channels_res = Resource(env, channels_no)  # number of channels available
        self.max_queue_sz = max_queue_sz  # max queue size
        self.requests_rt = requests_rt  # requests incoming ratio
        self.service_rt = service_rt  # requests processing ratio
        self.v_param = v_param  # exponential distribution parameter # v_param == beta == 1 / lambda
        self.stats = Statistics(self)


    @staticmethod
    def run(env: Environment, model: 'QSM'):
        request_id = 0
        while True:
            yield env.timeout(exponential(1.0 / model.requests_rt))
            env.process(QSM._start_request_lifecycle(env, request_id, model))
            request_id += 1


    def _awaiting(self, request_id: int):
        yield self.env.timeout(exponential(1.0 / self.v_param))


    def _processing(self, request_id: int):
        yield self.env.timeout(exponential(1.0 / self.service_rt))


    @staticmethod
    def _start_request_lifecycle(env: Environment, request_id: int, model: 'QSM'):
        model.stats.requests_awaiting.append(len(model.channels_res.queue))  # stats
        model.stats.requests_processing.append(model.channels_res.count)  # stats

        with model.channels_res.request() as request:
            cur_queue_length = len(model.channels_res.queue)

            if model.max_queue_sz != math.inf:
                """
                In case a queue is bounded...
                """
                if cur_queue_length <= model.max_queue_sz:
                    """
                    If there are available places in a queue, request is passed. 
                    """
                    start_tstamp = env.now

                    happened_events = yield request | env.process(model._awaiting(request_id))
                    model.stats.time_spent_awaiting.append(env.now - start_tstamp)  # stats

                    if request in happened_events:
                        yield env.process(model._processing(request_id))
                        model.stats.requests_completed_amount += 1  # stats
                    else:
                        model.stats.requests_rejected_amount += 1  # stats
                    model.stats.time_spent_total.append(env.now - start_tstamp)  # stats

                else:
                    """
                    If there are no places left in a queue, request is rejected.
                    """
                    model.stats.requests_rejected_amount += 1  # stats
            else:
                """
                In case a queue is unbounded, request is simply put in the queue and awaits 
                it's  time, then it's processed.
                """
                yield QSM._process_in_infinite_queue(env, request_id, request, model)


    @staticmethod
    def _process_in_infinite_queue(env: Environment
                                   , request_id: int
                                   , request: Request
                                   , model: 'QSM'):
        start_tstamp = env.now
        yield request
        model.stats.time_spent_awaiting.append(env.now - start_tstamp)  # stats
        yield env.process(model._processing(request_id))
        model.stats.time_spent_total.append(env.now - start_tstamp)  # stats
        model.stats.requests_completed_amount += 1  # stats
