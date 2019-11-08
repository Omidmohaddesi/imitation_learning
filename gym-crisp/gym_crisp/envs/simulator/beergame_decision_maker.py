"""Decision makers for beer game"""

from decision_maker import DecisionMaker
from decision_maker import allocate_equally
from decision import ProduceDecision, OrderDecision, TreatDecision


class BeerGameDecisionMaker(DecisionMaker):
    """Common helper functions for beer game decision makers"""

    def __init__(self, agent):
        """
        :param agent: The agent that the decision maker makes decision for
        :type agent: Agent
        """
        super(DecisionMaker, self).__init__()
        self.agent = agent

        self.lead_time = 3
        self.desired_inventory = 80
        self.stock_adjustment_time = 1
        self.weight_of_supply_line = 1

    def desired_supply_line(self):
        return self.agent.predicted_demand * self.lead_time

    def effective_inventory(self):
        return self.agent.inventory_level() - self.agent.backlog_level()

    def supply_line(self):
        return self.agent.on_order_level()

    def supply_line_adjustment(self):
        return self.weight_of_supply_line * \
               (self.desired_supply_line() - self.supply_line()) / \
               self.stock_adjustment_time

    def inventory_adjustment(self):
        return (self.desired_inventory - self.effective_inventory()) / \
               self.stock_adjustment_time

    def order_amount(self, allocated_inventory):
        order_amount = int(self.agent.predicted_demand
                           # - allocated_inventory
                           + self.inventory_adjustment()
                           + self.supply_line_adjustment())

        if order_amount < 0:
            order_amount = 0

        # print self.agent.name(), ":", self.agent.predicted_demand, allocated_inventory, \
        #     self.inventory_adjustment(), self.supply_line_adjustment(), ", ", \
        #     self.agent.inventory_level(), self.agent.backlog_level(), self.agent.on_order_level(), ", ", \
        #     order_amount

        return order_amount


class HealthCenterDecisionMaker(BeerGameDecisionMaker):
    """Beer game decision maker for health centers"""

    def __init__(self, hc):
        super(HealthCenterDecisionMaker, self).__init__(hc)

    def make_decision(self, now):
        self.agent.decisions = []
        total_inventory = self.agent.inventory_level()
        total_demand = self.agent.urgent + self.agent.non_urgent
        treat_decision = TreatDecision()
        if total_inventory < total_demand:
            treat_decision.non_urgent = int(
                total_inventory * (float(self.agent.non_urgent) / total_demand))
            treat_decision.urgent = total_inventory - treat_decision.non_urgent
            self.agent.decisions.append(treat_decision)
        else:
            treat_decision.non_urgent = self.agent.non_urgent
            treat_decision.urgent = self.agent.urgent
            self.agent.decisions.append(treat_decision)
        allocated = treat_decision.urgent + treat_decision.non_urgent

        order_amount = self.order_amount(allocated)
        num_upstream_node = len(self.agent.upstream_nodes)
        for agent in self.agent.upstream_nodes:
            order_decision = OrderDecision()
            order_decision.upstream = agent
            order_decision.amount = int(
                float(order_amount) / num_upstream_node)
            self.agent.decisions.append(order_decision)


class DistributorDecisionMaker(BeerGameDecisionMaker):
    """Beer game decision maker for distributors"""

    def __init__(self, ds):
        super(DistributorDecisionMaker, self).__init__(ds)

    def make_decision(self, now):
        self.agent.decisions = []
        allocated = allocate_equally(self.agent)

        order_amount = self.order_amount(allocated)
        num_upstream_nodes = len(self.agent.upstream_nodes)
        for agent in self.agent.upstream_nodes:
            decision = OrderDecision()
            decision.upstream = agent
            decision.amount = int(float(order_amount) / num_upstream_nodes)
            self.agent.decisions.append(decision)

class ManufacturerDecisionMaker(BeerGameDecisionMaker):
    """Beer game decision maker for manufacturers"""

    def __init__(self, mn):
        super(ManufacturerDecisionMaker, self).__init__(mn)

        self.lead_time = self.agent.lead_time

    def supply_line(self):
        return sum([item.amount for item in self.agent.in_production])

    def make_decision(self, now):
        self.agent.decisions = []
        allocated = allocate_equally(self.agent)

        decision = ProduceDecision()
        decision.amount = self.order_amount(allocated)
        self.agent.decisions.append(decision)
