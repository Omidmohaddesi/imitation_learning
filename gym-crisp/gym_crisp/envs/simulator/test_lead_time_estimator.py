import unittest
import numpy.testing as npt

from agent import Agent
from agent import Item
from lead_time_estimator import LeadTimeEstimator


class TestLeadTimeEstimator(unittest.TestCase):

    def test_estimate(self):
        agent = Agent()
        dst = Agent()
        agent.upstream_nodes.append(dst)
        estimator = LeadTimeEstimator(agent)

        # | Cycle | 10  | 11 | 12  | 13 | 14  | 15 | 16 |
        # | Order | 100 |    | 200 |    |     |    | 20 |
        # | Recv  |     | 10 |     |    | 120 |    | 60 |

        agent.make_order(dst, 100, 10)
        agent.make_order(dst, 200, 12)
        agent.make_order(dst, 20, 16)

        item1 = Item()
        item1.amount = 10
        agent.receive_delivery(item1, dst, 11)

        item2 = Item()
        item2.amount = 120
        agent.receive_delivery(item2, dst, 14)

        item3 = Item()
        item3.amount = 60
        agent.receive_delivery(item3, dst, 16)

        estimator.estimate(17)

        npt.assert_almost_equal(agent.on_time_delivery_rate[0], 40.0 / 300.0)
        npt.assert_almost_equal(agent.effective_lead_times[0], 3.875)
        npt.assert_almost_equal(agent.effective_lead_time, 3.875)



