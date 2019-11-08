import copy
import pandas as pd

from order import Order
from demand_predictor import *
from lead_time_estimator import LeadTimeEstimator
from order_up_to_level_calculator import *
from network import InTransit


class Agent(object):
    """Agent is an entity that participate in the supply chain"""

    def __init__(self):
        self.id = 0
        self.agent_type = "agent"
        self.agent_name = "agent"
        self.inventory = []
        self.predicted_demand = 0.0
        self.predicted_demand_stdev = 0.0
        self.up_to_level = 0.0
        # self.last_forecast = 0
        self.decisions = []
        self.upstream_nodes = []
        self.effective_lead_times = []
        self.effective_lead_time = 2.0
        self.on_time_delivery_rate = []
        self.downstream_nodes = []
        self.backlog = []
        self.on_order = []
        self.history = {}
        self.history_length = int(100)

        self.demand_predictor = None
        self.lead_time_estimator = None
        self.order_up_to_level_calculator = None
        self.lead_time_dict={}
        self.last_forecast=0

        # needed for PsychSim
        # self.lead_time_dict = {}
        self.in_transit_inventory = {}
        self.in_transit_inventory_2 = 40
        self.in_transit_inventory_2 = 40

    @staticmethod
    def new_history_item(time):
        return {
            'time': time,
            'inventory': 0,
            'allocate': [],
            'production': [],
            'delivery': [],
            'incoming_order': [],
            'patient': (0, 0),
            'patient_lost': (0, 0),
            'treat': (0, 0),
            'order': [],
        }

    def is_history_available(self, time):
        """checks if the history of a certain time is still available or not"""

        return time in self.history


    def get_history_item(self, time):
        """get_current_history_item returns the history item that records
        what is happening now. If such item is already in the agents history,
        this function will simply return it. If the history item has not been
        created yet, this function will create one and return it.

        :param time: the history item you want you retrieve
        :type time: int

        :return: The history item of the current time
        """

        if time not in self.history:
            history_item = self.new_history_item(time)
            self.history[time] = history_item
            return history_item

        history_item = self.history[time]
        return history_item

    def inventory_level(self):
        """ the number of units of drugs currently available in the inventory
        :return: current inventory level
        """

        level = 0

        for item in self.inventory:
            level += item.amount

        return level

    def backlog_level(self):
        """
        :return: The current number of drugs in backlog
        """
        return sum(order.amount for order in self.backlog)

    def demand(self, now):
        """
        get the demand of cycle now. The now time must be within the recorded
        history. If now is too old, this function returns 0.

        :param now: time
        :type now: int
        :return: the demand of cycle specified by now
        :rtype: int
        """
        item = self.get_history_item(now)
        demand = sum(in_order.amount for in_order in item['incoming_order'])
        return demand

    def on_order_level(self):
        """return the current on order level"""
        return sum(item.amount for item in self.on_order)

    def receive_delivery(self, package, src, time):
        """ receives a package delivery from src

            :param package: the drug to receive
            :param src: from which agent
            :param time: current time

            :type package: Item
            :type src: Agent
            :type time: int
        """
        self.update_order_according_to_delivery(package, src, time)
        self.inventory.append(copy.copy(package))
        self.record_delivery_history(package, src, time)

    def update_order_according_to_delivery(self, package, src, time):
        """ updates the order information upon receiving a package
        :param package: the package that is received
        :param src: the agent that send out the package
        :param time: the time that the package is received

        :type package: Item
        :type src: Agent
        :type time: int
        """
        amount = package.amount
        on_order_id = 0
        while amount > 0 and on_order_id < len(self.on_order):
            on_order = self.on_order[on_order_id]

            if on_order.dst != src:
                on_order_id += 1
                continue

            if amount < on_order.amount:
                self.mark_delivery_in_order_history(on_order, amount, time)
                on_order.amount -= amount
                amount = 0
            else:
                self.mark_delivery_in_order_history(on_order, on_order.amount, time)
                amount -= on_order.amount
                on_order.amount = 0

            on_order_id += 1

        if amount > 0:
            raise ValueError("Delivered amount more than on_order amount")

        self.on_order[:] = [o for o in self.on_order if not o.amount <= 0]

    def mark_delivery_in_order_history(self, on_order, amount, time):
        """ Associate the delivery history with the order history

            :param on_order: the order from the on_order list that received
                             the delivery
            :param amount: the amount of the delivery
            :param time: the time that the delivery happens

            :type on_order: Order
            :type amount: integer
            :type time: integer
        """
        history_item = self.get_history_item(on_order.place_time)

        # Search for the order that matches the on_order provided.
        # Since everything in the history is a copy of its original data, this
        # search is inevitable.
        matching_order = None
        for order in history_item['order']:
            if order.id == on_order.id:
                matching_order = order

        delivery = {'amount': amount, 'time': time}
        if matching_order is not None:
            matching_order.delivery.append(delivery)

    def record_delivery_history(self, package, src, time):
        """ record the delivery history
        :param package: the package received
        :param src: the agent that send out this package
        :param time: the time when the package is received
        """
        history_item = self.get_history_item(time)
        delivery = {
            'src': src,
            'item': copy.copy(package),
        }
        history_item['delivery'].append(delivery)

    def receive_order(self, order, now):
        """ receive_order puts the order in the backlog and record the
            history

            :param order: the order to receive
            :type order: Order
            :param now: current time
            :type now: int
        """
        self.backlog.append(copy.copy(order))

        history_item = self.get_history_item(now)
        history_item['incoming_order'].append(copy.copy(order))

    def make_order(self, dst, amount, now):
        """ makes an order to the destination

        :param dst: the destination agent
        :param amount: the amount to order
        :param now: current time
        :type dst: Agent
        :type amount: int
        :type now: int

        :return: the order made
        :rtype: Order
        """
        order = Order()
        order.amount = amount
        order.dst = dst
        order.src = self
        order.place_time = now

        self.on_order.append(copy.copy(order))

        history_item = self.get_history_item(now)
        history_item['order'].append(copy.copy(order))

        return order

    def update(self, now):
        """ Update agent status """
        self.remove_too_old_history(now)
        self.record_inventory_history(now)

        (self.predicted_demand, self.predicted_demand_stdev) = \
            self.demand_predictor.predict_demand(now)
        self.lead_time_estimator.estimate(now)
        self.up_to_level = self.order_up_to_level_calculator.calculate(now)

    def remove_too_old_history(self, now):
        """ removes the history items that is older than history_length cycles
            long

            :param now: current time
            :type now: int
        """
        self.history = {k:v for k,v in self.history.iteritems()
                           if k > now - self.history_length}

    def record_inventory_history(self, now):
        """ record the current inventory level in the history """
        history_item = self.get_history_item(now)
        history_item['inventory'] = self.inventory_level()

    def name(self):
        """
        returns the name of the agent.
        :return: name of the agent
        :rtype: string
        """
        return self.agent_name

    def fulfill_order_with_item(self, order, item, amount, network, time):
        shipment_item = copy.copy(item)
        shipment_item.amount = amount
        in_transit = InTransit(shipment_item)
        in_transit.leadTime = network.connectivity[self.id, order.src.id]
        in_transit.sendTime = time

        in_transit.src = self
        in_transit.dst = order.src
        network.payloads.append(in_transit)

        order.amount -= amount
        self.backlog[:] = [o for o in self.backlog if o.amount > 0]

        item.amount -= amount
        self.inventory[:] = [i for i in self.inventory if not i.amount <= 0]

        self._add_allocation_history(time, shipment_item, order)

    def _add_allocation_history(self, time, item, order):
        history_item = self.get_history_item(time)
        if 'allocate' not in history_item:
            history_item['allocate'] = []
        history_item['allocate'].append({
            'item': copy.copy(item),
            'order': copy.copy(order),
        })


    def collect_data(self, now):
        """
        Collect data collects data for data analytics and visualization
        :param now: current time
        :type now: int
        :return: a array of entries that contains all the collected data
        :rtype: list
        """
        name = self.name()

        return [
            [now, name, 'inventory', self.inventory_level() - self.backlog_level(), ''],
            [now, name, 'demand', self.demand(now), ''],
            # [now, name, 'predicted_demand', self.predicted_demand, ''],
            # [now, name, 'up-to-level', self.up_to_level, ''],
            # [now, name, 'eff-lead-time', self.effective_lead_time, 'cycle'],
            [now, name, 'on-order', self.on_order_level(), ''],
            # [now, name, 'shipment-out', sum([allocation['item'].amount for allocation in self.get_history_item(now)['allocate']])],
            [now, name, 'shipment-in', sum([allocation['item'].amount for allocation in self.get_history_item(now)['delivery']])],
            [now, name, 'order', sum([order.amount for order in self.get_history_item(now)['order']]), ]
        ]


class Item(object):
    """ Item represents the drug in the inventory or in-transit
    """
    def __init__(self):
        self.made_by = None
        self.made_time = 0
        self.line_no = 0
        self.batch_no = None
        self.amount = int(0)
        self.life_time = 10
        # For the in-production items
        self.lead_time = int(2)


class Manufacturer(Agent):

    def __init__(self):
        super(Manufacturer, self).__init__()

        self.agent_type = 'mn'

        # list of Agent
        self.downstream_nodes = []

        self.in_production = []
        self.line_capacity = 0
        self.num_active_lines = 0
        self.prod_amount = 0

        # Related to the in production drug
        self.lead_time = 2

    def update(self, now):
        super(Manufacturer, self).update(now)
        self.apply_in_production_in_history(now)

    def apply_in_production_in_history(self, now):
        for in_production in self.in_production:
            in_production.lead_time -= 1
            if in_production.lead_time <= 0:
                self.inventory.append(in_production)
                self.add_production_history(now, in_production)

        self.in_production[:] = [
            i for i in self.in_production if not i.lead_time <= 0]

    def add_production_history(self, now, item):
        history_item = self.get_history_item(now)
        if 'production' not in history_item:
            history_item['production'] = []
        history_item['production'].append(copy.copy(item))

    def collect_data(self, now):
        data = super(Manufacturer, self).collect_data(now)

        name = self.name()
        data.append([now, name, 'in_production',
                     sum(in_prod.amount for in_prod in self.in_production), ''])

        return data


class Distributor(Agent):
    def __init__(self):
        super(Distributor, self).__init__()

        self.agent_type = 'ds'

        # dictonary of percentage of on-time delivery from each upstream node
        self.ontime_deliv_rate = {}
        self.expctd_on_order = {}


class HealthCenter(Agent):
    def __init__(self):
        super(HealthCenter, self).__init__()
        self.agent_type = 'hc'
        # list of Order
        self.urgent = 0
        self.non_urgent = 0
        self.satisfied_urgent = 0
        self.satisfied_non_urgent = 0
        self.treat_threshold = 20

        # Dictionary of percentage of on-time delivery from each upstream node
        self.ontime_deliv_rate = {}
        self.expctd_on_order = {}

    def demand(self, now):
        """
        return the current demand of cycle now, which is the sum of the
        urgent and non-urgent patient.

        :param now: current time
        :type now: int
        :return: the demand of cycle now
        :rtype: int
        """
        history_item = self.get_history_item(now)
        patient = history_item['patient']
        return sum(patient)

    def receive_patient(self, urgent, non_urgent, now):
        """
        discard lost patient and update new patient amount

        :param urgent: urgent patient amount
        :type urgent: int
        :param non_urgent: non-urgent patient amount
        :type non_urgent: int
        :param now: current time
        :type now: int
        """
        history_item = self.get_history_item(now)
        history_item['patient_lost'] = (self.urgent, self.non_urgent)

        self.urgent = urgent
        self.non_urgent = non_urgent

        history_item['patient'] = (urgent, non_urgent)

    def backlog_level(self):
        return self.urgent + self.non_urgent

    def collect_data(self, now):
        data = super(HealthCenter, self).collect_data(now)

        name = self.name()
        data.append([now, name, 'lost', self.urgent + self.non_urgent, ''])

        return data


class AgentBuilder(object):
    """ AgentBuilder builds fully usable agent """

    def __init__(self):
        self.demand_predictor_type = "MovingAverage"
        self.cycle_service_level = 0.97
        self.lead_time = 2
        self.review_time = 1
        self.history_preserve_time = 100
        self.next_id = 0
        self.fixed_order_up_to_level = False

    def use_fixed_order_up_to_level(self):
        self.fixed_order_up_to_level = True

    def build(self, agent_type):
        """ build an agent

        :param agent_type: the type of the agent
        :type agent_type: string

        :return: the agent that is built
        """

        if agent_type == 'health_center':
            agent = HealthCenter()
        elif agent_type == 'distributor':
            agent = Distributor()
        elif agent_type == 'manufacturer':
            agent = Manufacturer()
        else:
            raise ValueError("agent type can only be health_center, "
                             "wholesaler, distributor, or manufacturer")

        agent.id = self.next_id
        self.next_id += 1

        if self.demand_predictor_type == 'MovingAverage':
            demand_predictor = MovingAverageDemandPredictor(
                agent,
                self.history_preserve_time
            )
        elif self.demand_predictor_type == 'RunningAverage':
            demand_predictor = RunningAverageDemandPredictor(agent)
        elif self.demand_predictor_type == 'ExponentialSmoothingDemand':
            demand_predictor = ExponentialSmoothingDemandPredictor(agent)
        else:
            raise ValueError("demand predictor type can only be MovingAverage "
                             "or RunningAverage")

        lead_time_estimator = LeadTimeEstimator(agent)
        agent.lead_time_estimator = lead_time_estimator

        if self.fixed_order_up_to_level:
            order_level_calculator = NullOrderUpToLevelCalculator(agent)
        else:
            order_level_calculator = OrderUpToLevelCalculatorImpl(agent)
            order_level_calculator.lead_time = self.lead_time
            order_level_calculator.review_period = self.review_time
            order_level_calculator.cycle_service_level = self.cycle_service_level

        agent.demand_predictor = demand_predictor
        agent.order_up_to_level_calculator = order_level_calculator
        agent.history_length = self.history_preserve_time

        return agent



