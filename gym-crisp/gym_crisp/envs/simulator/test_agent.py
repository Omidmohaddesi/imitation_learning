import unittest
from agent import *
from order import *


class TestAgent(unittest.TestCase):

    def test_receive_delivery(self):
        upstream_agent1 = Agent()
        upstream_agent2 = Agent()
        agent = Agent()

        order1 = Order()
        order1.amount = 10
        order1.src = agent
        order1.dst = upstream_agent1
        order1.place_time = 10
        agent.on_order.append(order1)
        agent.get_history_item(10)['order'].append(copy.copy(order1))

        order2 = Order()
        order2.amount = 20
        order2.src = agent
        order2.dst = upstream_agent2
        order2.place_time = 11
        agent.on_order.append(order2)
        agent.get_history_item(11)['order'].append(copy.copy(order2))

        order3 = Order()
        order3.amount = 30
        order3.src = agent
        order3.place_time = 12
        order3.dst = upstream_agent1
        agent.on_order.append(order3)
        agent.get_history_item(12)['order'].append(copy.copy(order3))

        item = Item()
        item.amount = 20
        agent.receive_delivery(item, upstream_agent1, 20)

        self.assertEqual(len(agent.on_order), 2)
        self.assertEqual(order1.amount, 0)
        self.assertEqual(order2.amount, 20)
        self.assertEqual(order3.amount, 20)

        self.assertEqual(agent.inventory[0].amount, 20)

        self.assertEqual(len(agent.get_history_item(20)['delivery']), 1)
        self.assertEqual(len(agent.get_history_item(10)['order'][0].delivery), 1)
        self.assertEqual(len(agent.get_history_item(12)['order'][0].delivery), 1)

    def test_remove_too_old_history(self):
        agent = Agent()
        agent.history_length = 50

        agent.get_history_item(0)
        agent.get_history_item(1)
        agent.get_history_item(100)
        agent.get_history_item(101)

        agent.remove_too_old_history(120)
        self.assertEqual(len(agent.history), 2)

    def test_record_inventory_history(self):
        agent = Agent()

        item1 = Item()
        item1.amount = 10
        agent.inventory.append(item1)

        item2 = Item()
        item2.amount = 15
        agent.inventory.append(item2)

        agent.record_inventory_history(50)

        self.assertEqual(agent.get_history_item(50)['inventory'], 25)

    def test_receive_order(self):
        agent = Agent()

        order = Order()
        order.amount = 10

        agent.receive_order(order, 100);

        self.assertEqual(len(agent.backlog), 1)
        self.assertEqual(agent.backlog[0].amount, 10)

        self.assertEqual(len(agent.get_history_item(100)['incoming_order']), 1)
        order_in_history = agent.get_history_item(100)['incoming_order'][0]
        self.assertEqual(order_in_history.amount, 10)

    def test_make_order(self):
        agent = Agent()
        dst = Agent()

        order = agent.make_order(dst, 100, 10)

        self.assertEqual(order.place_time, 10)
        self.assertEqual(order.amount, 100)
        self.assertEqual(order.src, agent)
        self.assertEqual(order.dst, dst)

        self.assertEqual(len(agent.on_order), 1)
        self.assertEqual(len(agent.get_history_item(10)['order']), 1)

    def test_demand(self):
        agent = Agent()

        order1 = Order()
        order1.amount = 20
        agent.receive_order(order1, 100)

        order2 = Order()
        order2.amount = 40
        agent.receive_order(order2, 100)

        demand = agent.demand(100)
        self.assertEqual(demand, 60)


class TestHealthCenter(unittest.TestCase):

    def test_demand(self):
        health_center = HealthCenter()
        health_center.receive_patient(10, 100, 50)

        demand = health_center.demand(50)
        self.assertEqual(demand, 110)

    def test_receive_patient(self):
        health_center = HealthCenter()
        health_center.urgent = 16
        health_center.non_urgent = 116

        health_center.receive_patient(10, 100, 150)

        self.assertEqual(health_center.urgent, 10)
        self.assertEqual(health_center.non_urgent, 100)

        history_item = health_center.get_history_item(150)
        self.assertEqual(history_item['patient'], (10, 100))
        self.assertEqual(history_item['patient_lost'], (16, 116))


class TestManufacturer(unittest.TestCase):

    def test_apply_in_production_in_history(self):

        manufacturer = Manufacturer()

        item1 = Item()
        item1.lead_time = 1
        item1.amount = 10
        manufacturer.in_production.append(item1)

        item2 = Item()
        item2.lead_time = 2
        item2.amount = 20
        manufacturer.in_production.append(item2)

        manufacturer.apply_in_production_in_history(10)

        self.assertEqual(manufacturer.inventory_level(), 10)
        self.assertEqual(len(manufacturer.get_history_item(10)['production']),
                         1)
        self.assertEqual(len(manufacturer.in_production), 1)

