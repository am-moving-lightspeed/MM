from math import factorial
from typing import List
from typing import Tuple

from numpy import add
from numpy import array
from numpy import prod


class Statistics:

    def __init__(self, model):
        self.model = model

        self.time_spent_awaiting = []  # Time spent awaiting in a queue

        self.time_spent_total = []  # Time spent awaiting in a queue and being processed

        self.requests_completed_amount = 0  # Amount of requests, that were completed

        self.requests_rejected_amount = 0  # Amount of requests, that were rejected

        self.requests_awaiting = []  # Amount of requests in a queue, when new request incoming

        self.requests_processing = []  # Amount of processing requests, when new request incoming


    def get_amounts_of_requests_in_system_at_time(self) -> List:
        return add(array(self.requests_awaiting), array(self.requests_processing)).tolist()


    def get_average_amount_of_requests_processed_at_time(self) -> float:
        return array(self.requests_processing).mean()


    def get_average_amount_of_requests_in_system_at_time(self) -> float:
        return array(self.get_amounts_of_requests_in_system_at_time()).mean()


    def get_average_amount_of_requests_in_queue_at_time(self) -> float:
        return array(self.requests_awaiting).mean()


    def get_average_time_of_request_spent_in_system(self) -> float:
        return array(self.time_spent_total).mean()


    def get_average_time_of_request_spent_in_queue(self) -> float:
        return array(self.time_spent_awaiting).mean()


    def find_empiric_probs(self) -> Tuple:
        requests_passed_total = self.requests_completed_amount + self.requests_rejected_amount

        """
        P -- array of probabilities. P[i] is the probability, that
        req_proc_and_await_at_time[i] == i.
        """
        P = []
        """
        req_proc_and_await_at_time -- vector, where each component shows how many
        requests there had been processing and in a queue by the time a new request came.
        """
        req_proc_and_await_at_time = add(
          array(self.requests_processing),
          array(self.requests_awaiting)
        )
        max_req_proc_and_await_amount = self.model.channels_no + self.model.max_queue_sz
        for i in range(1, max_req_proc_and_await_amount + 1):
            matches = req_proc_and_await_at_time[req_proc_and_await_at_time == i]
            P.append(len(matches) / requests_passed_total)

        """
        P_rejected -- probability for a request to be rejected.
        """
        P_rejected = self.requests_rejected_amount / requests_passed_total

        """
        Q -- probability for a request to be processed.
        """
        Q = 1 - P_rejected

        """
        A -- absolute throughput.
        """
        A = Q * self.model.requests_rt

        avg_req_in_system = self.get_average_amount_of_requests_in_system_at_time()
        avg_req_processed = self.get_average_amount_of_requests_processed_at_time()
        avg_req_in_queue = self.get_average_amount_of_requests_in_queue_at_time()
        avg_req_time_in_system = self.get_average_time_of_request_spent_in_system()
        avg_req_time_in_queue = self.get_average_time_of_request_spent_in_queue()

        return (P, Q, A, P_rejected, avg_req_in_system, avg_req_in_queue, avg_req_time_in_system,
                avg_req_time_in_queue, avg_req_processed)


    def find_theoretical_probs(self):
        ro = self.model.requests_rt / self.model.service_rt
        betta = self.model.v_param / self.model.service_rt

        """
        P -- array of probabilities. P[i] is the probability, that
        req_proc_and_await_at_time[i] == i.
        """
        P: list = []
        p00 = sum([ro ** i / factorial(i) for i in range(self.model.channels_no + 1)])
        p01 = (ro ** self.model.channels_no / factorial(self.model.channels_no))
        p02 = sum([
          ro ** i / (prod([self.model.channels_no + t * betta for t in range(1, i + 1)]))
          for i in range(1, self.model.max_queue_sz + 1)
        ])
        p0 = (p00 + p01 * p02) ** -1

        P.append(p0)
        for i in range(1, self.model.channels_no + 1):
            px = (ro ** i / factorial(i)) * p0
            P.append(px)

        pn = P[-1]
        for i in range(1, self.model.max_queue_sz):
            px = (ro ** i / prod([self.model.channels_no + t * betta for t in range(1, i + 1)])) \
                 * pn
            P.append(px)

        P_rejected = P[-1]

        Q = 1 - P[-1]

        A = Q * self.model.requests_rt

        avg_req_in_queue = sum([
          i * pn * (ro ** i) / prod([self.model.channels_no + t * betta for t in range(1, i + 1)])
          for i in range(1, self.model.max_queue_sz + 1)]
        )

        avg_req_in_system = sum([
          index * p0 * (ro ** index) / factorial(index)
          for index in range(1, self.model.channels_no + 1)
        ]) + sum([
          (self.model.channels_no + index) * pn * ro ** index
          / prod(array([self.model.channels_no + t * betta for t in range(1, index + 1)]))
          for index in range(1, self.model.max_queue_sz + 1)
        ])

        avg_req_processed = A / self.model.service_rt
        avg_req_time_in_system = avg_req_in_system / self.model.requests_rt
        avg_req_time_in_queue = Q * ro / self.model.requests_rt

        return (P, Q, A, P_rejected, avg_req_in_system, avg_req_in_queue, avg_req_time_in_system,
                avg_req_time_in_queue, avg_req_processed)
