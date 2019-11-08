
class Cost(object):

    def __init__(self):
        self.id = Order.new_id()

        self.src = None
        self.dst = None
        self.amount = int(0)
        self.place_time = 0

        self.delivery = []

        self.recv_time = 0
        self.exp_recv_time = 0
        self.expire_time = 0

    def TotalCost(self, now):

        inventoryCost = self.agent.unit_inventory_holding_cost
        backlogCost = self.agent.unit_backlog_cost

        totalCost = self.totalCost + (inventoryCost * self.inventory_level) + (backlogCost * self.backlog_level)

        return totalCost

# total_cost[hc, t] = total_cost[hc, t-1] +(unit_inventory_holding_cost*inventory[hc,t]+ unit_backlog_cost*backlog[hc,t])
# total_cost[ws, t] = total_cost[ws, t-1] +(unit_inventory_holding_cost*inventory[ws,t]+ unit_backlog_cost*backlog[ws,t])
# total_cost[ds, t] = total_cost[ds, t-1] +(unit_inventory_holding_cost*inventory[ds,t]+ unit_backlog_cost*backlog[ds,t])
# total_cost[mn, t] = total_cost[mn, t-1] +(unit_inventory_holding_cost*inventory[mn,t]+ unit_backlog_cost*backlog[mn,t])