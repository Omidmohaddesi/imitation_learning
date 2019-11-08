'''
########################## Parameters and initial values of the mathematical model


####  is (1/alphas) in Sterman for all four echelon  for i = R, W, D, F
stock_adjustment_time[i] = 1 # is a decision variable

#### for all four echelon  for i = R, W, D,
mailing_delay_time[i] = 1

#### for all four echelon  for i =  W, D, F
shipment_time[i] = 2

##### only for manufacturer
production_lead_time[i] = 2

### Beta in Sterman for i = R, W, D, F
weight_of_supply_line[i] = 1  # is a decision variable

### Theta in Sterman for i = R, W, D, F
smoothing_factor[i] = 0.2  # is a decision variable

####Normal demand
#### end_customer_demand = 40 for 4 weeks and then 80

if t < 5:
    end_customer_demand = 40
else:
    end_customer_demand = 80


########################## Parameters and initial values of the mathematical model_ initial inventory
#### Calculated by the hospiltal  the initial is 40
expected_end_customer_demand = 40
#### Calculated by the wholesalor, distributor and  factory  the initial is 40
expected_orders = 40

#### for i = R, W, D, F Optimal value
desired_inventory = 0  # is a decision variable

desired_supply_line[hc,0] = expected_end_customer_demand[0] *( mailing_delay_time[hc] + shipment_time[ws])
desired_supply_line[ws,0] = expected_orders[hc,0] *( mailing_delay_time[ws] + shipment_time[ds])
desired_supply_line[ds,0] = expected_orders[ws,0] *( mailing_delay_time[ds] + shipment_time[mn])
desired_supply_line[mn,0] = expected_orders[ds,0] *( production_lead_time)


########################## Parameters and initial values of the mathematical model_ initial backlogs

#### Initial value  for i = R, W, D, F
backlog = 0
inventory = 120

#### Initial value  for i = R, W, D
in_transit_inventory_2 = 40
in_transit_inventory_1 = 40

#### Initial value  for i =  F
work_in_process_inventories_1 = 40
work_in_process_inventories_2 = 40

#### Initial value for time= 1 for i = R, W, D
orders = 40
production_start_rate = 40
incoming_orders = 40

#### Initial value for time= 1 for i = R, W, D, f
total_cost = 0

unit_inventory_holding_cost = 5 # this amount in sterman's paper is 0.5
unit_backlog_cost = 10 # this amount in sterman's paper is 0


############ The remaining model equations are given in an order based on the steps of
############ the game presented in Sterman (1989). This sequence should strictly be followed while
############ performing calculations to ensure an accurate representation of the board version of The Beer Game
############ low from right to left (i.e., from the upper echelon to the lower)
############ and orders flow from left to right (i.e., from lower echelon to the upper)

######### Step 1. Receive inventory and advance shipping delays

## for i = R, W, D

inventory[i,t] = inventory[i,t-1] + in_transit_inventory_2[i,t-1]

## for i = F
inventory[mn,t] = inventory[mn,t-1] +  work_in_process_inventories_2[t-1]

######  for i = R, W, D
in_transit_inventory_2[i,t] = in_transit_inventory_1[i,t-1]

work_in_process_inventories_2[t] = work_in_process_inventories_1[t-1]

###### for i = R, W, D
in_transit_inventory_1[i,t] = 0

work_in_process_inventories_1[t] = 0


######## Step 2. Fill orders

if inventory[hc,t] <= (backlog[hc, t-1] + end_customer_demand[t]):
    shipments[pa,t] = inventory[hc,t]
else:
    shipments[pa, t] = backlog[hc, t-1] + end_customer_demand[t]


if inventory[ws,t] <= (backlog[ws, t-1] + incoming_orders[ws, t]):
    shipments[hc,t] = inventory[ws,t]
else:
    shipments[hc, t] = backlog[ws, t-1] + incoming_orders[ws, t]

if inventory[ds,t] <= (backlog[ds, t-1] + incoming_orders[ds, t]):
    shipments[ws,t] = inventory[ds,t]
else:
    shipments[ws, t] = backlog[ds, t-1] + incoming_orders[ds, t]

if inventory[mn,t] <= (backlog[mn, t-1] + incoming_orders[mn, t]):
    shipments[ds,t] = inventory[mn,t]
else:
    shipments[ds, t] = backlog[mn, t-1] + incoming_orders[mn, t]

### for i = R, W, D

in_transit_inventory_1[i,t] = shipments[i,t]

work_in_process_inventories_1[t] = production_start_rate[t]


######## Step 3. Record inventory or backlog

backlog[hc,t] = backlog[hc,t-1] + end_customer_demand[t] - shipments[pa,t]

backlog[ws,t] = backlog[ws,t-1] + incoming_orders[ws,t] - shipments[hc,t]

backlog[ds,t] = backlog[ds,t-1] + incoming_orders[ds,t] - shipments[ws,t]

backlog[mn,t] = backlog[mn,t-1] + incoming_orders[mn,t] - shipments[ws,t]


inventory[hc, t] = inventory[hc,t] -  shipments[pa,t]

inventory[ws, t] = inventory[ws,t] -  shipments[hc,t]

inventory[ds, t] = inventory[ds,t] -  shipments[ws,t]

inventory[mn, t] = inventory[mn,t] -  shipments[ds,t]


### Expectation formation simple exponential smoothing

expected_end_customer_demand[t] = expected_end_customer_demand[t-1] + smoothing_factor[hc] * (end_customer_demand[t]-end_customer_demand[t-1])

expected_orders[hc, t] = expected_orders[hc, t-1] + smoothing_factor[ws] *(incoming_orders[ws,t]-expected_orders[hc,t-1])

expected_orders[ws, t] = expected_orders[ws, t-1] + smoothing_factor[ds] *(incoming_orders[ds,t]-expected_orders[ws,t-1])

expected_orders[ds, t] = expected_orders[ds, t-1] + smoothing_factor[mn] *(incoming_orders[mn,t]-expected_orders[ds,t-1])

######## Step 4. Advance the order slips


incoming_orders[ws, t+1] = orders[hc, t]

incoming_orders[ds, t+1] = orders[ws, t]

incoming_orders[mn, t+1] = orders[ds, t]




######## Step 5. Place orders

desired_supply_line[hc, t] = expected_end_customer_demand[t]*(mailing_delay_time[hc] + shipment_time[ws])

desired_supply_line[ws, t] = expected_orders[hc, t]*(mailing_delay_time[ws] + shipment_time[ds])

desired_supply_line[ds, t] = expected_orders[ws, t]*(mailing_delay_time[ds] + shipment_time[mn])

desired_supply_line[mn, t] = expected_orders[ds, t]* production_lead_time


effective_inventory[hc, t] = inventory[hc,t] - backlog[hc,t]

effective_inventory[ws, t] = inventory[ws,t] - backlog[ws,t]

effective_inventory[ds, t] = inventory[ds,t] - backlog[ds,t]

effective_inventory[mn, t] = inventory[mn,t] - backlog[mn,t]


supply_line[hc, t]= incoming_orders[ws, t+1] + backlog[ws,t] + in_transit_inventory_1[hc, t] + in_transit_inventory_2[hc,t]

supply_line[ws, t]= incoming_orders[ds, t+1] + backlog[ds,t] + in_transit_inventory_1[ws, t] + in_transit_inventory_2[ws,t]

supply_line[ds, t]= incoming_orders[mn, t+1] + backlog[mn,t] + in_transit_inventory_1[ds, t] + in_transit_inventory_2[ds,t]

supply_line[mn, t]= work_in_process_inventories_1[t] + work_in_process_inventories_2[t]


supply_line_adjustment[hc, t] = weight_of_supply_line[hc]*(desired_supply_line[hc, t] - supply_line[hc, t]) /stock_adjustment_time[hc]
supply_line_adjustment[ws, t] = weight_of_supply_line[ws]*(desired_supply_line[ws, t] - supply_line[ws, t]) /stock_adjustment_time[ws]
supply_line_adjustment[ds, t] = weight_of_supply_line[ds]*(desired_supply_line[ds, t] - supply_line[ds, t]) /stock_adjustment_time[ds]
supply_line_adjustment[mn, t] = weight_of_supply_line[mn]*(desired_supply_line[mn, t] - supply_line[mn, t]) /stock_adjustment_time[mn]


inventory_adjustment[hc, t] = (desired_inventory[hc] - effective_inventory[hc, t])/ stock_adjustment_time[hc]
inventory_adjustment[ws, t] = (desired_inventory[ws] - effective_inventory[ws, t])/ stock_adjustment_time[ws]
inventory_adjustment[ds, t] = (desired_inventory[ds] - effective_inventory[ds, t])/ stock_adjustment_time[ds]
inventory_adjustment[mn, t] = (desired_inventory[mn] - effective_inventory[mn, t])/ stock_adjustment_time[mn]

############ ordering rate; you can use this formula or players order

if t < 5:
    orders[hc, t + 1] = 4
else:
    orders[hc, t+1] = max(0, int(expected_end_customer_demand[t]+inventory_adjustment[hc, t] + supply_line_adjustment[hc, t]))

if t < 5:
    orders[ws, t + 1] = 4
else:
    orders[ws, t+1] = max(0, int(expected_orders[hc, t]+inventory_adjustment[ws, t] + supply_line_adjustment[ws, t]))

if t < 5:
    orders[ds, t + 1] = 4
else:
    orders[ds, t+1] = max(0, int(expected_orders[ws, t]+inventory_adjustment[ds, t] + supply_line_adjustment[ds, t]))

if t < 5:
    production_start_rate[mn, t + 1] = 4
else:
    production_start_rate[mn, t+1] = max(0, int(expected_orders[ds, t]+inventory_adjustment[mn, t] + supply_line_adjustment[mn, t]))


############ Cost function

total_cost[hc, t] = total_cost[hc, t-1] +(unit_inventory_holding_cost*inventory[hc,t]+ unit_backlog_cost*backlog[hc,t])
total_cost[ws, t] = total_cost[ws, t-1] +(unit_inventory_holding_cost*inventory[ws,t]+ unit_backlog_cost*backlog[ws,t])
total_cost[ds, t] = total_cost[ds, t-1] +(unit_inventory_holding_cost*inventory[ds,t]+ unit_backlog_cost*backlog[ds,t])
total_cost[mn, t] = total_cost[mn, t-1] +(unit_inventory_holding_cost*inventory[mn,t]+ unit_backlog_cost*backlog[mn,t])

'''