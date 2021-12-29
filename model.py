import math
from typing import Tuple, List

import numpy

from agent import Agent
from matrix import Matrix


class Bush_Mosteller:
    def __init__(self, agents: tuple[Agent], game: Matrix, payoffs: Tuple[int]):
        """
        :param agents: A list of agents
        :param game: The game to be played
        :param payoffs: A vector of payoffs
        """
        self.agents = agents
        self.game = game
        self.payoffs = payoffs
        self.actions, self.action_probabilities, self.aspirations, self.stimuli = None, None, None, None

    def update_agent_aspirations(self, payoffs: List[float]) -> List[float]:
        """
        Update the aspirations of the agents
        :param payoffs: List of payoffs where payoff[i] is the payoff received by agent i
        :return: A list of aspirations where the ith element corresponds to the aspiration of agent i
        """
        return [agent.updt_aspi(payoffs[i]) for i, agent in enumerate(self.agents)]

    def compute_stimuli(self, payoffs: Tuple[int]) -> numpy.ndarray:
        """
        Computes the stimuli for all agents
        :param payoffs: List of payoffs where payoff[i] is the payoff received by agent i
        :return: A list of stimuli where the ith element corresponds to the stimulus for agent i
        """
        stimuli = numpy.zeros(len(self.agents))
        for i, payoff in enumerate(payoffs):
            aspiration = self.agents[i].get_aspiration()
            sup = math.ceil(max(payoff - aspiration for payoff in self.payoffs))
            stimuli[i] = (payoff - aspiration) / sup
        return stimuli

    def get_agent_actions(self) -> List[int]:
        """
        Queries all agents' next action
        :return: A list of actions where the ith element corresponds to action taken by agent i
        """
        return [agent.act() for agent in self.agents]

    def run_episode(self) -> tuple[list[int], list[list[int]], numpy.ndarray, list[float]]:
        """
        Runs a single episode of the game
        :return: A tuple containing: The actions taken by the agents,
        The probabilities of the actions, the stimuli and the aspirations respectively
        """
        actions = self.get_agent_actions()
        payoffs = self.game.get_payoff(actions)
        stimuli = self.compute_stimuli(payoffs)
        aspirations = self.update_agent_aspirations(payoffs)

        action_probabilities = [[0, 0] for _ in range(len(self.agents))]
        for i, agent in enumerate(self.agents):
            action_probabilities[i][actions[i]] = agent.learn(stimuli[i], actions[i])
            action_probabilities[i][1 - actions[i]] = 1 - action_probabilities[i][actions[i]]

        return actions, action_probabilities, stimuli, aspirations

    def run(self, nb_runs: int) -> None:
        self.actions = numpy.zeros(nb_runs, dtype=List[int, int])
        self.aspirations = numpy.zeros(nb_runs, dtype=float)
        self.action_probabilities = numpy.zeros(nb_runs, dtype=List[float, float])
        self.stimuli = numpy.zeros(nb_runs, dtype=List[float, float])

        for i in range(nb_runs):

            self.actions[i], self.action_probabilities[i], self.stimuli[i], self.aspirations[i] = self.run_episode()

    def get_actions(self) -> Tuple[Tuple[int]]:
        """
        :return: Tuple containing tuples where the ith tuple contains all the actions of agent i during training
        """
        return tuple(zip(*self.actions))

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
