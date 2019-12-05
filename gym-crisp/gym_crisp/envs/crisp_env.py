import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from .simulator import simulation_builder
from .simulator.decision import TreatDecision
from .simulator.decision import OrderDecision
from .simulator.decision import ProduceDecision
from .simulator.decision import AllocateDecision

import logging
logger = logging.getLogger(__name__)


class CrispEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=20, role='ws'):
        self.n = n  # number of simulation periods in each episode
        self.simulation = None
        self.runner = None
        self.state = None
        self.agent = None
        self.decisions = []
        self.role = role    # the role of the agent
        self.backlog = 0
        self.reward = 0
        self.order = 0
        self.total_reward = 0
        self.min_order = 0
        self.max_order = 10000
        self.action_space = spaces.Box(
            low=self.min_order, high=self.max_order, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([self.max_order, self.max_order, self.max_order, self.max_order]),
            dtype=np.float32)

        self.seed()
        self.reset()

    def step(self, action):

        reward = self.reward    # in this version reward only depends on allocation and inventory and not the #
        # agent's action
        self._take_action(action)
        self.runner._make_decision(self.simulation.now)
        self.agent = self._get_agent_by_role(self.role)
        self.agent.decisions = []
        self._parse_decisions()
        self.runner._apply_decision(self.simulation.now)

        # draw_figures(game, user_id, agent.name())

        self.simulation.now += 1
        self.runner._update_patient(self.simulation.now)
        self.runner._update_network(self.simulation.now)
        self.runner._update_agents(self.simulation.now)
        self.runner._exogenous_event(self.simulation.now)

        done = 0
        new_obs = self._get_obs(self.simulation.now, self.backlog)

        return new_obs, reward, done, {'time': self.simulation.now}

    def reset(self):
        self.simulation, self.runner = simulation_builder.build_simulation_beer_game()
        self.agent = self._get_agent_by_role(self.role)

        self.runner._update_patient(0)
        self.runner._update_agents(0)
        self.simulation.now += 1
        self.runner._update_patient(self.simulation.now)
        self.runner._update_network(self.simulation.now)
        self.runner._update_agents(self.simulation.now)
        self.runner._exogenous_event(self.simulation.now)

        self.backlog = 0

        return self._get_obs(self.simulation.now, self.backlog)

    def render(self, mode='human'):
        # Render the environment to the screen

        self.total_reward += self.reward
        print(f'Simulation Time: {self.simulation.now}')
        print(f'Inventory: {self.agent.inventory_level()}')
        print(f'Backlog: {self.backlog}')
        print(f'Order amount: {self.order}')
        print(f'Reward: {self.reward}')
        print(f'total reward: {self.total_reward}')
        print('--------------------')

    def _reward(self, inventory, allocation, backlog):

        inventory_cost = int((inventory - allocation) / 2)
        backlog_cost = int(backlog)

        self.reward = -(inventory_cost + backlog_cost)
        return self.reward

    def _get_obs(self, now, backlog):

        inventory = self.agent.inventory_level()
        history_item = self.agent.get_history_item(now)
        shipment = sum(d['item'].amount for d in history_item['delivery'])
        demand = self.agent.demand(now)

        if backlog + demand >= inventory:
            allocation = inventory
            backlog += demand - inventory
        else:
            allocation = backlog + demand
            backlog = 0

        decision = {
            'agent': self.agent,
            'decision_name': 'satisfy_hc1',
            'decision_value': int(allocation),
        }

        self._reward(inventory, allocation, backlog)

        self.backlog = backlog
        self.decisions.append(decision)

        return np.array([inventory, shipment, demand, backlog])

    def _get_agent_by_role(self, role):
        #   IMPORTANT:  This part probably needs to be edited. Right now this function only returns the first agent in
        #               the list from simulation but in a more complex network it should check to see which
        #               agent is gonna be the RL agent and return that one.

        agent_list = None

        if role == 'hc':
            agent_list = self.simulation.health_centers
        elif role == 'ws':
            agent_list = [k for k in self.simulation.distributors if k.agent_name == 'WS']
        elif role == 'ds':
            agent_list = [k for k in self.simulation.distributors if k.agent_name == 'DS']
        elif role == 'mn':
            agent_list = self.simulation.manufacturers

        return agent_list[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _take_action(self, action):

        decision = {
            'agent': self.agent,
            'decision_name': 'order_from_ds1',
            'decision_value': int(action),
        }

        self.decisions.append(decision)
        self.order = action

    def _parse_decisions(self):
        """ convert the game decision to simulator desicion """
        for decision in self.decisions:
            self._convert_to_simulation_decision(decision)

        self.decisions = []

    def _convert_to_simulation_decision(self, agent_decision):
        """
        :type agent_decision: RL Agent Decision
        """

        if agent_decision['decision_name'] == 'satisfy_urgent':
            decision = TreatDecision()
            decision.urgent = agent_decision['decision_value']
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_non_urgent':
            decision = TreatDecision()
            decision.non_urgent = agent_decision['decision_value']
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_ds1':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            # decision.upstream = self.simulation.distributors[0]
            decision.upstream = [k for k in self.simulation.distributors if k.agent_name == 'DS'][0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_ds2':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            # decision.upstream = self.simulation.distributors[1]
            decision.upstream = [k for k in self.simulation.distributors if k.agent_name == 'DS'][1]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_mn1':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            decision.upstream = self.simulation.manufacturers[0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_mn2':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            decision.upstream = self.simulation.manufacturers[1]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'produce':
            decision = ProduceDecision()
            decision.amount = agent_decision['decision_value']
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_ds1':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.distributors[0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_ds2':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.distributors[1]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_hc1':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.health_centers[0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_hc2':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.health_centers[1]
            agent_decision['agent'].decisions.append(decision)

        else:
            print("Decision type " + agent_decision['decision_name']
                  + " not supported!\n")
            return
