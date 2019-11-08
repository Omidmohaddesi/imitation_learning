import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .simulator import agent, simulation_builder

# import logging
# logger = logging.getLogger(__name__)


class CrispEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=20, role='WS'):
        self.n = n  # number of simulation periods in each episode
        self.simulation = None
        self.runner = None
        self.state = None
        self.role = role    # the role of the agent
        self.min_order = 0
        self.max_order = 10000
        self.action_space = spaces.Box(
            low=self.min_order, high=self.max_order, shape=(1,), dtype=np.int64)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([self.max_order, self.max_order, self.max_order, self.max_order]),
            dtype=np.int64)

    def step(self, action):
        return obs, -cost, done, {}

    def reset(self):
        self.simulation, self.runner = simulation_builder.build_simulation_beer_game()
        self.runner._update_patient(0)
        self.runner._update_agents(0)
        self.simulation.now += 1
        self.runner._update_patient(self.simulation.now)
        self.runner._update_network(self.simulation.now)
        self.runner._update_agents(self.simulation.now)
        self.runner._exogenous_event(self.simulation.now)

        return self._get_obs()

    def _get_obs(self):
        return np.array([inventory, shipment, demand, backlog])

