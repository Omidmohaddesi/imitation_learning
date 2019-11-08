"""SimulationEncoder provides tools to encode a simulation
"""

import json
import agent
import network
import simulation

class SimulationJsonEncoder(object):
    """ SimulationJsonEncoder encodes the simulation in JSON format
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def encode(self):
        """ return the JSON representation of the simulation in a string
        :return: the JSON representation
        """
        return json.dumps(self.simulation, indent=2, cls=JsonEncoder)

class JsonEncoder(json.JSONEncoder):
    """The JSON Encoder that replaced the default JSON encoder"""
    def default(self, o):

        if isinstance(o, agent.Agent):
            ret = o.id
        elif isinstance(o, network.Network):
            ret = {
                'num_agents': o.num_agent,
            }
        elif isinstance(o, simulation.Simulation):
            ret = {
                'now': o.now,
                'health_centers': [hc.__dict__  for hc in o.health_centers],
                'wholesaler': [ws.__dict__ for ws in o.wholesalers],
                'distributors': [ds.__dict__  for ds in o.distributors],
                'manufacturers': [mn.__dict__  for mn in o.manufacturers],
                'network': o.network,
                'info_network': o.info_network,
                'disruptions': o.disruptions,
            }
        else:
            ret = o.__dict__

        return ret
