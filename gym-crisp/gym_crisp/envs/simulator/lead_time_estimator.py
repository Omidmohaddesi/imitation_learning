""" lead_time_estimator provides tools for estimating effective lead time and
    on-time delivery rate
"""


class LeadTimeEstimator(object):

    def __init__(self, agent):
        """
        Constructor
        :param agent: the agent that the LeadTimeEstimator is working on
        :type agent: Agent
        """
        self.agent = agent
        self.grace_period = 2  # todo this should be 2? # 3  # order at cycle 0, receive at cycle 3, OK
        self.trust_hist_length = 20
    def estimate(self, now):
        """ estimates the effective lead time and the on-time deliver rate """
        self.estimate_on_time_delivery_rate(now)
        self.estimate_effective_lead_time(now)
        self.estimate_overall_effective_lead_time(now)

    def estimate_on_time_delivery_rate(self, now):
        self.agent.on_time_delivery_rate = []

        for node in self.agent.upstream_nodes:
            delivered = 0.0
            ordered = 0.0

            for _, history_item in self.agent.history.items():

                # Too recent history
                if history_item['time'] < 10 or history_item['time'] < now - self.trust_hist_length or \
                                history_item['time'] >= now - self.grace_period:
                    continue

                for order in history_item['order']:
                    if order.dst != node:
                        continue

                    ordered += order.amount
                    for delivery in order.delivery:
                        if delivery['time'] <= order.place_time + self.grace_period:
                            delivered += delivery['amount']

            if ordered == 0:
                self.agent.on_time_delivery_rate.append(1)
            else:
                # print 'ordered'
                # print ordered
                # print 'delivered'
                # print delivered
                # print 'rate'
                # print delivered / ordered
                self.agent.on_time_delivery_rate.append(delivered / ordered)

    def estimate_effective_lead_time(self, now):
        self.agent.effective_lead_times = []
        for node in self.agent.upstream_nodes:
            delivered = 0.0
            ordered = 0.0

            for _, history_item in self.agent.history.items():

                for order in history_item['order']:
                    delivered_in_order = 0

                    if order.dst != node:
                        continue

                    ordered += order.amount
                    for delivery in order.delivery:
                        delivered += delivery['amount'] * (delivery['time'] - order.place_time)
                        delivered_in_order += delivery['amount']

                    remaining = order.amount - delivered_in_order
                    delivered += remaining * (now - order.place_time)

            if ordered == 0:
                self.agent.effective_lead_times.append(self.grace_period)
            else:
                self.agent.effective_lead_times.append(delivered / ordered)

    def estimate_overall_effective_lead_time(self, now):
        delivered = 0.0
        ordered = 0.0

        for _, history_item in self.agent.history.items():

            for order in history_item['order']:
                delivered_in_order = 0

                ordered += order.amount
                for delivery in order.delivery:
                    delivered += delivery['amount'] * (delivery['time'] - order.place_time)
                    delivered_in_order += delivery['amount']

                remaining = order.amount - delivered_in_order
                delivered += remaining * (now - order.place_time)

        if ordered == 0:
            self.agent.effective_lead_time = self.grace_period
        else:
            self.agent.effective_lead_time = delivered / ordered
