import math
from typing import Tuple, List

import numpy
from numpy import ndarray

from agent import Agent
from matrix_payoffs import Matrix_Payoffs


class Bush_Mosteller:
    def __init__(self, agents: List[Agent], game: Matrix_Payoffs):
        """
        :param agents: A list of agents
        :param game: The game to be played
        """
        self.agents = agents
        self.game = game
        self.payoffs = game.get_payoffs_vector()
        self.action_probabilities, self.aspirations, self.stimuli = None, None, None

    def update_agent_aspirations(self, rewards: Tuple[int]) -> List[float]:
        """
        Update the aspirations of the agents
        :param rewards: List of payoffs where payoff[i] is the payoff received by agent i
        :return: A list of aspirations where the ith element corresponds to the aspiration of agent i
        """
        return [agent.updt_aspi(rewards[i]) for i, agent in enumerate(self.agents)]

    def compute_stimuli(self, rewards: Tuple[int]) -> List[float]:
        """
        Computes the stimuli for all agents
        :param rewards: List of payoffs where payoff[i] is the payoff received by agent i
        :return: A list of stimuli where the ith element corresponds to the stimulus for agent i
        """
        supremi = [self.get_supremum(agent.aspi) for agent in self.agents]

        return [agent.cpt_stimuli(rewards[i], supremi[i]) for i, agent in enumerate(self.agents)]

    def get_supremum(self, aspi):
        return math.ceil(max(abs(self.payoffs[0] - aspi), abs(self.payoffs[1] - aspi), abs(self.payoffs[2] - aspi),
                             abs(self.payoffs[3] - aspi)))

    def query_next_actions(self) -> List[int]:
        """
        Queries all agents' next action
        :return: A list of actions where the ith element corresponds to action taken by agent i
        """
        return [agent.act() for agent in self.agents]

    def run_episode(self) -> Tuple[ndarray, List[float], List[float]]:
        """
        Runs a single episode of the game
        :return: A tuple containing: The actions taken by the agents,
        The probabilities of the actions, the stimuli and the aspirations respectively
        """
        actions = self.query_next_actions()
        payoffs = self.game.get_payoff(actions)
        stimuli = self.compute_stimuli(payoffs)
        aspirations = self.update_agent_aspirations(payoffs)

        action_probabilities = numpy.zeros(shape=(len(self.agents), 2), dtype=numpy.float64)
        for i, agent in enumerate(self.agents):
            action_probabilities[i] = agent.learn(stimuli[i], actions[i])

        return action_probabilities, stimuli, aspirations

    def run(self, nb_runs: int) -> None:
        self.aspirations = numpy.empty(nb_runs, dtype=object)
        self.action_probabilities = numpy.empty(nb_runs, dtype=object)
        self.stimuli = numpy.empty(nb_runs, dtype=object)

        for i in range(nb_runs):
            self.action_probabilities[i], self.stimuli[i], self.aspirations[i] = self.run_episode()

    def get_aspirations(self) -> Tuple[Tuple[float]]:
        """
        :return: Tuple containing tuples where the ith tuple contains all the aspirations of agent i during training
        """
        return tuple(zip(*self.aspirations))

    def get_action_probabilities(self) -> Tuple[Tuple[List[float]]]:
        """
        :return: Tuple containing tuples of lists where the ith tuple contains all the action probabilities of agent
        i during training as lists
        """
        return tuple(zip(*self.action_probabilities))

    def get_stimuli_agent(self) -> Tuple[Tuple[float]]:
        """
        :return: Tuple containing tuples where the ith tuple contains all the stimuli of agent i during training
        """
        return tuple(zip(*self.stimuli))
