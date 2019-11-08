"""simulation_builder provides utility classes to build games"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import simulator.simulation as sim
import simulator.agent as agent
import simulator.simulation_runner as sim_runner
import simulator.decision_maker as dmaker
import simulator.beergame_decision_maker as bg_dmaker
import simulator.patient_model as pmodel
import simulator.distruption as disr
from simulator.agent import Item
from simulator.network import Network, InTransit, OrderMessage
from simulator.order import Order
import simulator.network as network
matplotlib.use('agg')


def build_simulation():
    """ return a new instance of the simulation """
    simulation = sim.Simulation()

    agent_builder = agent.AgentBuilder()
    agent_builder.lead_time = 2
    agent_builder.review_time = 0
    agent_builder.cycle_service_level = 0.9
    agent_builder.history_preserve_time = 20

    mn1 = agent_builder.build("manufacturer")
    mn1.num_of_lines = 20
    mn1.line_capacity = 20
    mn1.num_active_lines = 20

    mn2 = agent_builder.build("manufacturer")
    mn2.num_of_lines = 20
    mn2.line_capacity = 20
    mn2.num_active_lines = 20

    ds1 = agent_builder.build("distributor")
    ds2 = agent_builder.build("distributor")

    hc1 = agent_builder.build("health_center")
    hc2 = agent_builder.build("health_center")

    hc1.upstream_nodes.extend([ds1, ds2])
    hc2.upstream_nodes.extend([ds1, ds2])
    ds1.upstream_nodes.extend([mn1, mn2])
    ds1.downstream_nodes.extend([hc1, hc2])
    ds2.upstream_nodes.extend([mn1, mn2])
    ds2.downstream_nodes.extend([hc1, hc2])
    mn1.downstream_nodes.extend([ds1, ds2])
    mn2.downstream_nodes.extend([ds1, ds2])

    simulation.add_agent(mn1)
    simulation.add_agent(mn2)
    simulation.add_agent(ds1)
    simulation.add_agent(ds2)
    simulation.add_agent(hc1)
    simulation.add_agent(hc2)

    net = network.Network(6)
    info_net = network.Network(6)
    for i in range(6):
        for j in range(6):
            net.connectivity[i, j] = 1
            info_net.connectivity[i, j] = 0
    simulation.network = net
    simulation.info_network = info_net

    decision_maker = dmaker.PerAgentDecisionMaker()

    hc1_dmaker = dmaker.UrgentFirstHCDecisionMaker(hc1)
    decision_maker.add_decision_maker(hc1_dmaker)

    hc2_dmaker = dmaker.UrgentFirstHCDecisionMaker(hc2)
    decision_maker.add_decision_maker(hc2_dmaker)

    ds1_dmaker = dmaker.SimpleDSDecisionMaker(ds1)
    decision_maker.add_decision_maker(ds1_dmaker)

    ds2_dmaker = dmaker.SimpleDSDecisionMaker(ds2)
    decision_maker.add_decision_maker(ds2_dmaker)

    mn1_dmaker = dmaker.SimpleMNDecisionMaker(mn1)
    decision_maker.add_decision_maker(mn1_dmaker)

    mn2_dmaker = dmaker.SimpleMNDecisionMaker(mn2)
    decision_maker.add_decision_maker(mn2_dmaker)

    hc1.up_to_level = 279
    hc2.up_to_level = 279
    ds1.up_to_level = 140
    ds2.up_to_level = 404
    mn1.up_to_level = 202
    mn2.up_to_level = 329

    patient_model = pmodel.NormalDistPatientModel([hc1, hc2])
    simulation.patient_model = patient_model

    runner = sim_runner.SimulationRunner(simulation, decision_maker)

    return (simulation, runner)


def build_simulation_beer_game_oul():
    """ return a new instance of the simulation for the beer game network with order-up-to-level """
    simulation = sim.Simulation()
    mn, ds, ws, hc = config_agents(simulation)
    config_network(simulation)
    config_init_inventory(simulation)
    config_init_order(simulation)
    config_init_transit_shipment(simulation)
    config_init_work_in_progress_production(simulation)

    decision_maker = dmaker.PerAgentDecisionMaker()
    hc_dmaker = dmaker.UrgentFirstHCDecisionMaker(hc)
    ws_dmaker = dmaker.SimpleDSDecisionMaker(ws)
    ds_dmaker = dmaker.SimpleDSDecisionMaker(ds)
    mn_dmaker = dmaker.SimpleMNDecisionMaker(mn)
    decision_maker.add_decision_maker(hc_dmaker)
    decision_maker.add_decision_maker(ws_dmaker)
    decision_maker.add_decision_maker(ds_dmaker)
    decision_maker.add_decision_maker(mn_dmaker)

    patient_model = pmodel.ConstantPatientModel([hc])
    patient_model.urgent = 40
    patient_model.non_urgent = 0
    simulation.patient_model = patient_model

    disruption = disr.DemandChangeDisruption([patient_model])
    disruption.start_time = 4
    disruption.end_time = 100000000
    disruption.change = 40
    simulation.disruptions.append(disruption)

    hc.up_to_level = 176
    ws.up_to_level = 176
    ds.up_to_level = 176
    mn.up_to_level = 176

    runner = sim_runner.SimulationRunner(simulation, decision_maker)

    return (simulation, runner)


def build_simulation_beer_game():
    """ return a new instance of the simulation for the beer game study"""
    simulation = sim.Simulation()
    mn, ds, ws, hc = config_agents(simulation)
    config_network(simulation)
    config_init_inventory(simulation)
    config_init_order(simulation)
    config_init_transit_shipment(simulation)
    config_init_work_in_progress_production(simulation)

    decision_maker = dmaker.PerAgentDecisionMaker()
    hc_dmaker = bg_dmaker.HealthCenterDecisionMaker(hc)
    ws_dmaker = bg_dmaker.DistributorDecisionMaker(ws)
    ds_dmaker = bg_dmaker.DistributorDecisionMaker(ds)
    mn_dmaker = bg_dmaker.ManufacturerDecisionMaker(mn)
    decision_maker.add_decision_maker(hc_dmaker)
    decision_maker.add_decision_maker(ws_dmaker)
    decision_maker.add_decision_maker(ds_dmaker)
    decision_maker.add_decision_maker(mn_dmaker)

    patient_model = pmodel.ConstantPatientModel([hc])
    patient_model.urgent = 40
    patient_model.non_urgent = 0
    simulation.patient_model = patient_model

    disruption = disr.DemandChangeDisruption([patient_model])
    disruption.start_time = 4
    disruption.end_time = 100000000
    disruption.change = 40
    simulation.disruptions.append(disruption)

    # constant value
    # agent.stock_adjustment_time = 1
    # agent.mailing_delay_time = 1
    # agent.shipment_time = 2 # for 3 main MN, Dc,
    # agent.production_lead_time = 2 # only for mn
    # agent.weight_of_supply_line = 1
    # agent.smoothing_factor = 0.2
    #
    #
    #
    # agent.expected_orders = 40
    #
    # agent.desired_inventory = 0
    #
    # #### for i = hc, ws, ds, mn  Optimal value
    # agent.desired_inventory = 0  # is a decision variable
    # agent.predict_demand = 40
    #
    #
    #
    # ########################## Parameters and initial values of the mathematical model_ initial backlogs
    #
    # #### Initial value  for i = hc, ws, ds, mn
    # agent.backlog_level = 0
    # agent.inventory_level = 120
    #
    # #### Initial value  for i = hc, ws, ds
    # agent.in_transit_inventory_2 = 40
    # agent.in_transit_inventory_1 = 40
    # agent.demand = 40
    #
    # if time < 5:
    #     agent.order_amount = 40
    #     agent.prod_amount = 40
    #
    # #### Initial value  for i =  mn
    # agent.work_in_process_inventories_1 = 40
    # agent.work_in_process_inventories_2 = 40
    #
    # #### Initial value for time= 1 for i = hc, ws, ds
    # agent.orders = 40
    # agent.production_start_rate = 40
    # agent.incoming_orders = 40
    #
    # #### Initial value for time= 1 for i = hc, ws, ds, mn
    # agent.total_cost = 0
    #
    # agent.unit_inventory_holding_cost = 5  # this amount in sterman's paper is 0.5
    # agent.unit_backlog_cost = 10  # this amount in sterman's paper is 0

    runner = sim_runner.SimulationRunner(simulation, decision_maker)

    return (simulation, runner)


def config_network(simulation):
    p_net = Network(4)
    i_net = Network(4)
    for i in range(4):
        for j in range(4):
            p_net.connectivity[i, j] = 2
            i_net.connectivity[i, j] = 1
    simulation.network = p_net
    simulation.info_network = i_net


def config_init_inventory(simulation):
    for agent in simulation.agents:
        item = Item()
        item.amount = 120
        agent.inventory.append(item)


def config_init_backlog(simulation):
    pass


def config_init_work_in_progress_production(simulation):
    item = Item()
    item.amount = 40
    item.lead_time = 2
    simulation.agents[0].in_production.append(item)

    item = Item()
    item.amount = 40
    item.lead_time = 1
    simulation.agents[0].in_production.append(item)


def config_init_order(simulation):
    order = Order()
    order.src = simulation.agents[3]
    order.dst = simulation.agents[2]
    order.amount = 80
    simulation.agents[3].on_order.append(order)

    order = Order()
    order.src = simulation.agents[2]
    order.dst = simulation.agents[1]
    order.amount = 80
    simulation.agents[2].on_order.append(order)

    order = Order()
    order.src = simulation.agents[1]
    order.dst = simulation.agents[0]
    order.amount = 80
    simulation.agents[1].on_order.append(order)

    order = Order()
    order.src = simulation.agents[3]
    order.dst = simulation.agents[2]
    order.amount = 40
    simulation.agents[3].on_order.append(order)
    message = OrderMessage(order)
    message.leadTime = 1
    message.src = order.src
    message.dst = order.dst
    simulation.info_network.payloads.append(message)

    order = Order()
    order.src = simulation.agents[2]
    order.dst = simulation.agents[1]
    order.amount = 40
    simulation.agents[2].on_order.append(order)
    message = OrderMessage(order)
    message.leadTime = 1
    message.src = order.src
    message.dst = order.dst
    simulation.info_network.payloads.append(message)

    order = Order()
    order.src = simulation.agents[1]
    order.dst = simulation.agents[0]
    order.amount = 40
    simulation.agents[1].on_order.append(order)
    message = OrderMessage(order)
    message.leadTime = 1
    message.src = order.src
    message.dst = order.dst
    simulation.info_network.payloads.append(message)


def config_init_transit_shipment(simulation):
    item = Item()
    item.amount = 40
    in_transit = InTransit(item)
    in_transit.src = simulation.agents[0]
    in_transit.dst = simulation.agents[1]
    in_transit.leadTime = 2
    simulation.network.payloads.append(in_transit)

    item = Item()
    item.amount = 40
    in_transit = InTransit(item)
    in_transit.src = simulation.agents[0]
    in_transit.dst = simulation.agents[1]
    in_transit.leadTime = 1
    simulation.network.payloads.append(in_transit)

    item = Item()
    item.amount = 40
    in_transit = InTransit(item)
    in_transit.src = simulation.agents[1]
    in_transit.dst = simulation.agents[2]
    in_transit.leadTime = 2
    simulation.network.payloads.append(in_transit)

    item = Item()
    item.amount = 40
    in_transit = InTransit(item)
    in_transit.src = simulation.agents[1]
    in_transit.dst = simulation.agents[2]
    in_transit.leadTime = 1
    simulation.network.payloads.append(in_transit)

    item = Item()
    item.amount = 40
    in_transit = InTransit(item)
    in_transit.src = simulation.agents[2]
    in_transit.dst = simulation.agents[3]
    in_transit.leadTime = 2
    simulation.network.payloads.append(in_transit)

    item = Item()
    item.amount = 40
    in_transit = InTransit(item)
    in_transit.src = simulation.agents[2]
    in_transit.dst = simulation.agents[3]
    in_transit.leadTime = 1
    simulation.network.payloads.append(in_transit)


def config_agents(simulation):
    agent_builder = agent.AgentBuilder()
    agent_builder.lead_time = 3
    agent_builder.review_time = 0
    agent_builder.cycle_service_level = 0.9
    agent_builder.history_preserve_time = 20
    agent_builder.demand_predictor_type = "ExponentialSmoothingDemand"

    manufacture = agent_builder.build("manufacturer")
    manufacture.agent_name = "MN"
    manufacture.line_capacity = 1000
    manufacture.num_active_lines = 1
    manufacture.num_of_lines = 1
    manufacture.demand_predictor.mean = 40
    manufacture.demand_predictor.predicted_demand = 40
    distributor = agent_builder.build("distributor")
    distributor.agent_name = "DS"
    distributor.demand_predictor.mean = 40
    distributor.demand_predictor.predicted_demand = 40
    wholesaler = agent_builder.build("distributor")
    wholesaler.agent_name = "WS"
    wholesaler.demand_predictor.mean = 440
    wholesaler.demand_predictor.predicted_demand = 40
    health_center = agent_builder.build("health_center")
    health_center.agent_name = "HC"
    health_center.demand_predictor.mean = 40
    health_center.demand_predictor.predicted_demand = 40

    health_center.upstream_nodes.extend([wholesaler])
    wholesaler.upstream_nodes.extend([distributor])
    distributor.upstream_nodes.extend([manufacture])

    manufacture.downstream_nodes.extend([distributor])
    distributor.downstream_nodes.extend([wholesaler])
    wholesaler.downstream_nodes.extend([health_center])

    simulation.add_agent(manufacture)
    simulation.add_agent(distributor)
    simulation.add_agent(wholesaler)
    simulation.add_agent(health_center)

    return manufacture, distributor, wholesaler, health_center
