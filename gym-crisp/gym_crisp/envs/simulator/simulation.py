"""simulation package provides data structures to build a simulation"""

from agent import *


class Simulation(object):
    """ A Simulation represents the states of the everything being simulated.
    """

    def __init__(self):
        self.now = 0
        self.health_centers = []
        self.wholesalers = []
        self.distributors = []
        self.manufacturers = []
        self.agents = []  # A collection of all types of agents
        self.regulatory_agency = None
        self.network = None
        self.info_network = None
        self.patient_model = None
        self.disruptions = []

    def add_agent(self, agent):
        self.agents.append(agent)
        if isinstance(agent, HealthCenter):
            self.health_centers.append(agent)
        elif isinstance(agent, Distributor):
            self.distributors.append(agent)
        elif isinstance(agent, Manufacturer):
            self.manufacturers.append(agent)
