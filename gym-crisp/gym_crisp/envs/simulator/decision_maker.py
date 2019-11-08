''' decision_maker provides simple decision makers for the simulation '''

import copy as copy
from agent import *
from decision import *


class DecisionMaker(object):
    ''' Decision Maker makes decisions for all the agents '''

    def make_decision(self, now):
        ''' make decisions for all the agents'''
        pass


class PerAgentDecisionMaker(DecisionMaker):
    ''' PerAgentDecisionMaker is a decision maker that combines all the
        agent level decision makers.
    '''

    def __init__(self):
        self.decision_makers = []

    def add_decision_maker(self, decision_maker):
        ''' Register a agent level decision maker'''
        self.decision_makers.append(decision_maker)

    def make_decision(self, now):
        for decision_maker in self.decision_makers:
            decision_maker.make_decision(now)


class UrgentFirstHCDecisionMaker(DecisionMaker):
    def __init__(self, hc):
        self.hc = hc

    def make_decision(self, now):
        self.hc.decisions = []
        num_upstream_node = len(self.hc.upstream_nodes)
        total_inventory = self.hc.inventory_level()
        total_on_order = self.hc.on_order_level()
        total_demand = self.hc.urgent + self.hc.non_urgent

        # Treat decision
        treat_decision = TreatDecision()
        if total_inventory < total_demand:
            treat_decision.non_urgent = int(
                total_inventory * (float(self.hc.non_urgent) / total_demand))
            treat_decision.urgent = total_inventory - treat_decision.non_urgent
            self.hc.decisions.append(treat_decision)
        else:
            treat_decision.non_urgent = self.hc.non_urgent
            treat_decision.urgent = self.hc.urgent
            self.hc.decisions.append(treat_decision)

        # Order decision
        total_inventory = total_inventory - \
            treat_decision.urgent - treat_decision.non_urgent
        order_amount = self.hc.up_to_level - total_inventory - total_on_order + 70

        if order_amount < 0:
            order_amount = 0
        for agent in self.hc.upstream_nodes:
            order_decision = OrderDecision()
            order_decision.upstream = agent
            order_decision.amount = int(
                float(order_amount) / num_upstream_node)
            self.hc.decisions.append(order_decision)


class SimpleHCDecisionMaker(DecisionMaker):
    def __init__(self, hc):
        self.hc = hc

    def make_decision(self, now):
        # todo update trust

        self.hc.decisions = []
        totalInv = sum(tempIn.amount for tempIn in self.hc.inventory)
        totalOnOrder = sum(tempO.amount for tempO in self.hc.on_order)
        totalDemand = self.hc.urgent + self.hc.non_urgent

        decisionT = TreatDecision()
        if totalInv < totalDemand:
            decisionT.non_urgent = int(
                totalInv * (float(self.hc.non_urgent) / totalDemand))
            decisionT.urgent = int(
                totalInv * (float(self.hc.urgent) / totalDemand))
            self.hc.decisions.append(decisionT)
        else:
            decisionT.non_urgent = int(self.hc.non_urgent)
            decisionT.urgent = int(self.hc.urgent)
            self.hc.decisions.append(decisionT)

        # update residual inventory used later for calculating order amount
        self.res_inventory = totalInv - decisionT.urgent - decisionT.non_urgent

        # order decision
        # orderAmount = max(self.hc.up_to_level -
        #                   totalOnOrder - self.res_inventory + 70, 0)
        orderAmount = max(self.hc.up_to_level -
                          totalOnOrder - self.res_inventory, 0)
        # Order Equally Recipe
        noUptr = len(self.hc.upstream_nodes)
        for agent in self.hc.upstream_nodes:
            decisionO = OrderDecision()
            decisionO.upstream = agent
            decisionO.amount = int(float(orderAmount) / noUptr)
            self.hc.decisions.append(decisionO)
        yyyyy = 0

        # todo Order By Trust


class SimpleDSDecisionMaker(DecisionMaker):
    def __init__(self, ds):
        self.ds = ds

    def make_decision(self, now):
        self.ds.decisions = []
        inventory = self.ds.inventory_level()

        # allocated=allocate_proportional (self.ds)
        allocated = allocate_equally(self.ds)
        inventory -= allocated


        # Order Decision
        on_order = sum(tempO.amount for tempO in self.ds.on_order)
        backlog = self.ds.backlog_level()
        backlog -= allocated
        order_amount = max(self.ds.up_to_level + backlog -
                           on_order - inventory, 0)

        # Order Equally Recipe
        num_upstream_nodes = len(self.ds.upstream_nodes)
        for agent in self.ds.upstream_nodes:
            decision = OrderDecision()
            decision.upstream = agent
            decision.amount = int(float(order_amount) / num_upstream_nodes)
            self.ds.decisions.append(decision)

        # todo Order By Trust


class SimpleMNDecisionMaker(DecisionMaker):
    def __init__(self, mn):
        self.mn = mn

    def make_decision(self, now):
        self.mn.decisions = []
        inventory = self.mn.inventory_level()

        # each time choose one of these to test the recipe
        # allocated = allocate_proportional(self.mn)
        allocated = allocate_equally(self.mn)
        inventory -= allocated

        # production Decision
        available_capacity = self.mn.num_active_lines * self.mn.line_capacity
        in_production = sum(
            in_prod.amount for in_prod in self.mn.in_production)
        in_backlog = self.mn.backlog_level()
        in_backlog -= allocated
        prod_amount = self.mn.up_to_level + in_backlog - inventory - in_production
        if prod_amount < 0:
            prod_amount = 0
        elif prod_amount > available_capacity:
            prod_amount = available_capacity

        # line_cap = self.mn.line_capacity
        # if prod_amount > line_cap:
        #     prod_amount = (int(prod_amount - 1) / int(line_cap) + 1) * line_cap
        self.mn.prod_amount = prod_amount

        decision_p = ProduceDecision()
        decision_p.amount = prod_amount
        self.mn.decisions.append(decision_p)


def allocate_proportional(agent):
    backlog = agent.backlog_level()
    inventory = agent.inventory_level()
    allocated = 0
    if inventory == 0 or backlog == 0:
        return 0

    demand_of = {}
    for a in agent.downstream_nodes:
        demand_of[a] = 0

    for bl in agent.backlog:
        if not bl.src in demand_of:
            demand_of[bl.src] = 0
        demand_of[bl.src] += bl.amount

    allocate_to = {}
    for ag in agent.downstream_nodes:
        allocate_to[ag] = min(demand_of[ag],
                              (float(demand_of[ag] * inventory) / backlog))

    allocated = sum(allocate_to.values())

    inv_ptr = 0
    inv_left = agent.inventory[0].amount
    for bl in agent.backlog:
        bl_left = bl.amount
        while allocate_to[bl.src] > 0 and bl_left > 0:
            if min(allocate_to[bl.src], bl_left) <= inv_left:
                decision_al = AllocateDecision()
                decision_al.amount = min(allocate_to[bl.src], bl_left)
                allocate_to[bl.src] -= decision_al.amount
                bl_left -= decision_al.amount
                inv_left -= decision_al.amount
                decision_al.item = agent.inventory[inv_ptr]
                decision_al.order = bl
                agent.decisions.append(decision_al)
            else:
                decision_al = AllocateDecision()
                decision_al.amount = inv_left
                allocate_to[bl.src] -= decision_al.amount
                bl_left -= decision_al.amount
                decision_al.item = agent.inventory[inv_ptr]
                decision_al.order = bl
                agent.decisions.append(decision_al)
                inv_ptr += 1
                inv_left = agent.inventory[inv_ptr].amount

    return allocated


def allocate_equally(agent):
    inventory = agent.inventory_level()
    backlog = agent.backlog_level()
    allocated = 0

    if inventory == 0 or backlog == 0:
        return 0

    num_downstream = len(agent.downstream_nodes)
    demand_of = {}
    for a in agent.downstream_nodes:
        demand_of[a] = 0

    for bl in agent.backlog:
        if not bl.src in demand_of:
            demand_of[bl.src] = 0
        demand_of[bl.src] += bl.amount

    allocate_to = {}
    for ag in agent.downstream_nodes:
        allocate_to[ag] = min(
            demand_of[ag], (inventory/ num_downstream))

    allocated = sum(allocate_to.values())

    inv_ptr = 0
    inv_left = agent.inventory[0].amount
    for bl in agent.backlog:
        bl_left = bl.amount
        while allocate_to[bl.src] > 0 and bl_left > 0:
            if min(allocate_to[bl.src], bl_left) <= inv_left:
                decision_al = AllocateDecision()
                decision_al.amount = min(allocate_to[bl.src], bl_left)
                allocate_to[bl.src] -= decision_al.amount
                bl_left -= decision_al.amount
                inv_left -= decision_al.amount
                decision_al.item = agent.inventory[inv_ptr]
                decision_al.order = bl
                agent.decisions.append(decision_al)
            else:
                decision_al = AllocateDecision()
                decision_al.amount = inv_left
                allocate_to[bl.src] -= decision_al.amount
                bl_left -= decision_al.amount
                decision_al.item = agent.inventory[inv_ptr]
                decision_al.order = bl
                agent.decisions.append(decision_al)
                inv_ptr += 1
                inv_left = agent.inventory[inv_ptr].amount

    return allocated
