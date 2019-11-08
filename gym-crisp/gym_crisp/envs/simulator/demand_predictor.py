""" demand_predictor provide services to predict demands """

import numpy as np


class DemandPredictor(object):
    """ DemandPredictor is the general interface for all types for demand predictors """

    def predict_demand(self, now):
        """ calculate and returns the predicted demand

            :param now: current time
            :type now: int

            :return: a tuple, the first number is mean, the second element is
            standard deviation
        """
        pass


class RunningAverageDemandPredictor(DemandPredictor):
    """ RunningAverageDemandPredictor """

    def __init__(self, agent):
        """ Constructor
        :param agent: The agent that this demand predictor is working with.
        :type agent: Agent
        """
        self.agent = agent
        self.mean = 0.0
        self.stdev = 0.0
        self.started = False
        self.alpha = 0.1

    def predict_demand(self, now):
        demand = self.agent.demand(now)
        if not self.started:
            self.mean = demand
            self.stdev = 0
            self.started = True
        else:
            sdev = self.stdev * self.stdev
            sdev = (1 - self.alpha) * (sdev + self.alpha * (demand - self.mean) * (demand - self.mean))
            self.stdev = sdev ** 0.5
            self.mean = \
                (1 - self.alpha) * self.mean + \
                self.alpha *demand
        return self.mean, self.stdev


class MovingAverageDemandPredictor(DemandPredictor):
    """ MovingAverageDemandPredictor """

    def __init__(self, agent, window_width):
        """ Constructor
        :param agent: The agent that this demand predictor is working with.
        :type agent: Agent
        """
        self.agent = agent
        self.width = window_width

    def predict_demand(self, now):
        demand = []
        for i in range(now - self.width + 1, now + 1):
            if i < 0: continue;
            demand.append(self.agent.demand(i))

        return np.mean(demand), np.std(demand, ddof=1)  # ddof for sample stdev


class ExponentialSmoothingDemandPredictor(DemandPredictor):
    """ ExponentialSmoothingDemandPredictor """

    def __init__(self, agent):
        """ Constructor
        :param agent: The agent that this demand predictor is working with.
        :type agent: Agent
        """
        self.agent = agent
        self.smoothing_factor = 0.2
        self.stdev = 1
        self.mean = 0
        self.predicted_demand = 0

    def predict_demand(self, now):
        demand = self.agent.demand(now)
        self.predicted_demand = self.predicted_demand + self.smoothing_factor * (demand - self.predicted_demand)
        return self.predicted_demand, self.stdev
