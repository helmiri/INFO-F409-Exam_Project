import math
from typing import List, Tuple
import numpy as np


# Runner needs this

class Matrix_Payoffs:
    def __init__(self, payoffs, game):
        self.num_agents = 2
        if (game == "PD"):
            # Prisonner's Dilemma : T > R > P > S
            self.R = payoffs[1]
            self.T = payoffs[0]
            self.S = payoffs[3]
            self.P = payoffs[2]
        elif (game == "SG"):
            # Stag Hunt Game : R > T > P > S
            self.R = payoffs[0]
            self.T = payoffs[1]
            self.S = payoffs[3]
            self.P = payoffs[2]
        else:
            # Chicken Game: T > R > S > P
            self.R = payoffs[1]
            self.T = payoffs[0]
            self.S = payoffs[2]
            self.P = payoffs[3]
        self.vector = [self.R, self.T, self.S, self.P]
        self.payoff = np.array([self.R, self.T, self.S, self.P]).reshape(2, 2)

    def get_payoff(self, actions: List[int]) -> Tuple[int]:
        """
        Method to get a payoff value based on an action
        :param actions: the actions played
        :return: an integer
        """
        return self.payoff[actions[0]][actions[1]]

    def get_payoffs_vector(self) -> List[int]:
        """
        Method to get the payoffs vector of the game
        :return: a vector of integers
        """
        return self.vector

    def get_supremum(self, value):
        return math.ceil(max(abs(self.T - value), abs(self.R - value), abs(self.P - value), abs(self.S - value), ))
