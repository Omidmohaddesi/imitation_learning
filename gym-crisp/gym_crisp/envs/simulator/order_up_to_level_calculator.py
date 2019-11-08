""" provide services to calculate order-up-to-level """

import numpy as np
import scipy.stats as st


class OrderUpToLevelCalculator(object):
    """ OrderUpToLevelCalculator calculates the order-up-to-level """

    def calculate(self, now):
        """
        calculates the order-up-to-level
        :param now: current time
        :return: int
        """
        pass


class NullOrderUpToLevelCalculator(OrderUpToLevelCalculator):

    def __init__(self, agent):
        """
        constructor
        :param agent: The agent that the calculator works with
        :type agent: Agent
        """
        self.agent = agent

    def calculate(self, now):
        """ The NullOrderUpToLevelCalculator does not change the order up to level """
        return int(self.agent.up_to_level)


class OrderUpToLevelCalculatorImpl(OrderUpToLevelCalculator):

    def __init__(self, agent):
        """
        Constructor
        :param agent: the agent that this calculator is working with
        :type agent: Agent
        """
        self.review_period = 1
        self.lead_time = 1
        self.cycle_service_level = 0.97
        self.agent = agent

    def calculate(self, now):
        z = st.norm.ppf(self.cycle_service_level)
        mean = self.agent.predicted_demand
        stdev = self.agent.predicted_demand_stdev

        if np.isnan(mean):
            mean = 0

        if np.isnan(stdev):
            stdev = 0

        # lead_time = self.agent.effective_lead_time
        lead_time = self.lead_time
        level = (lead_time + self.review_period) * mean + \
            z * stdev * ((lead_time + self.review_period) ** 0.5)

        return int(level)
