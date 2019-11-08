import unittest
import numpy.testing as npt

from agent import *
from demand_predictor import *


class MockAgent(Agent):

    def __init__(self):
        super(MockAgent, self).__init__()
        self.demand_data = []

    def demand(self, now):
        return self.demand_data[now]


class TestRunningAverageDemandPredictor(unittest.TestCase):

    def test_predict_demand(self):
        agent = MockAgent()
        agent.demand_data = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        demand_predictor = RunningAverageDemandPredictor(agent)

        demand_predictor.alpha = 0.1
        demand = demand_predictor.predict_demand(0)
        self.assertEqual(demand[0], 10, "The first mean")
        self.assertEqual(demand[1], 0, "The first std deviation")

        demand = demand_predictor.predict_demand(1)
        self.assertEqual(demand[0], 9.1, "running mean")
        self.assertEqual(demand[1], 2.7, "running stdev")


class TestMovingAverageDemandPredictor(unittest.TestCase):

    def test_predict_demand(self):
        agent = MockAgent()
        agent.demand_data = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        demand_predictor = MovingAverageDemandPredictor(agent, 10)

        demand = demand_predictor.predict_demand(9)
        self.assertEqual(demand[0], 5.5, "mean")
        npt.assert_almost_equal(demand[1], 3.0276, 3)

