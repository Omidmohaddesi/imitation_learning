from decision_maker import DecisionMaker
from psychsim.world import World
from psychsim_agents.psychsim_distributor import PSDistributorAgent
from psychsim_agents.psychsim_health_center import PSHealthCenterAgent
from psychsim_agents.helper_functions import *


class PsychSimDecisionMaker(DecisionMaker):
    """
    PsychSimDecisionMaker is a decision maker that uses PsychSim to make decisions, i.e., provides wrappers for the
    simulator to call PsychSim to make decisions for all agents.
    """

    def __init__(self, world, agents):
        """
        :param World world: the PsychSim world.
        :param list agents: the list of PsychSimAgent to update to/from the simulation.
        """
        self.world = world
        self.agents = agents
        self.recipes_chosen = {}
        self.action_values = {}  # stores the actions' values for each agent
        self.rewards = {}  # stores the current reward / cost for each agent
        self.planning = {}  # stores the planning states for each agent
        self.trust = {}  # stores the current trust values for each agent

    def make_decision(self, now):

        # updates PsychSim from simulation values
        for agent in self.agents:
            agent.update_psychsim_from_simulation(now)

        # make PsychSim agents decide
        outcomes_1 = self.world.step(real=True)
        # self.world.explain(outcomes_1, level=3)
        # decision_infos_1 = get_decision_info_conseq_decision(outcomes_1, 'DS 1', 'cleaning')
        # explain_decisions_conseq_decision('DS 1', decision_infos_1)
        # decision_infos_2 = get_decision_info_conseq_decision(outcomes_1, 'DS 2', 'cleaning')
        # explain_decisions_conseq_decision('DS 2', decision_infos_2)


        # updates simulation (decisions) based on PsychSim actions/outcomes
        outcome_dict_1 = outcomes_1[0]['new'].domain()[0]

        # collects action chosen info
        for name, action in outcomes_1[0]['actions'].items():
            self.recipes_chosen[name] = action['verb']

        # collects reward info
        for agent in self.agents:
            self.rewards[agent.ps_agent.name] = agent.ps_agent.reward(self.world.state[None].domain()[0])

        # collects trust info
        for agent in self.agents:
            if hasattr(agent, 'trust'):
                self.trust[agent.ps_agent.name] = {}
                for u in agent.up_stream_agents:
                    self.trust[agent.ps_agent.name][u.ps_agent.name] = self.world.getValue(agent.trust[u])

        # collects decision info
        for name, decision in outcomes_1[0]['decisions'].items():
            self.action_values[name] = {}
            if 'V' not in decision:
                # agent does not have values (only had one action available)
                self.action_values[name][str(decision['action'])] = .0
            else:
                self.planning[name] = {}
                for action, action_info in decision['V'].items():

                    # gets values of all actions
                    for key, info in action_info.items():
                        if key == '__EV__':
                            self.action_values[name][str(action)] = float(info)
                        else:
                            # gets planning decisions
                            self.planning[name][action] = []
                            projection = info['projection']

                            # stores old state
                            horiz_info = []
                            horiz_state_info = {}
                            effect = info['state']
                            for feat_name, feat_value in effect.items():
                                horiz_state_info[feat_name] = feat_value
                            horiz_info.append(horiz_state_info)
                            self.planning[name][action].append(horiz_info)

                            while len(projection) > 0:
                                horiz_info = []

                                # collects effect of actions in state
                                horiz_state_info = {}
                                effect = projection[0]['state']
                                for feat_name, feat_value in effect.items():
                                    horiz_state_info[feat_name] = feat_value
                                horiz_info.append(horiz_state_info)

                                # collects actions planned for all agents
                                horiz_action_info = {}
                                actions = projection[0]['actions']
                                for ag_name, ag_action in actions.items():
                                    horiz_action_info[ag_name] = str(next(iter(action)))
                                horiz_info.append(horiz_action_info)

                                # moves to next planning horizon
                                self.planning[name][action].append(horiz_info)
                                projection = projection[0]['projection']

        for agent in self.agents:
            agent.update_simulation_from_psychsim(outcome_dict_1)

        # make cleaning agent decide
        outcomes_2 = self.world.step(real=True)
        # self.world.explain(outcomes_2, level=2)
        outcome_dict_2 = outcomes_2[0]['new'].domain()[0]
