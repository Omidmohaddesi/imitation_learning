import unittest
import numpy.testing as npt

from agent import *
from order_up_to_level_calculator import *


class TestRunningAverageDemandPredictor(unittest.TestCase):

    def test_calculate(self):
        agent = Agent()
        order_up_to_level_calculator = OrderUpToLevelCalculatorImpl(agent)
        order_up_to_level_calculator.review_period = 1
        order_up_to_level_calculator.lead_time = 2
        order_up_to_level_calculator.cycle_service_level = 0.97

        agent.effective_lead_time = 2
        agent.predicted_demand = 10
        agent.predicted_demand_stdev = 5

        level = order_up_to_level_calculator.calculate(10)

        npt.assert_almost_equal(level, 46, 2)
