from typing import List, Tuple

# Payoff vector indices
T = 0
R = 1
P = 2
S = 3

class Matrix_Payoffs:
    def __init__(self, payoffs):
        self.num_agents = 2
        self.vector = payoffs
        self.matrix = [[[payoffs[R]] * 2, [payoffs[S], payoffs[T]]],
                       [[payoffs[T], payoffs[S]], [payoffs[P]] * 2]]

    def get_payoff(self, actions: List[int]) -> Tuple[int]:
        """
        :param actions: the actions of the agents
        :return: A tuple where the ith element is agent i's reward
        """
        return tuple(self.matrix[actions[0]][actions[1]])

    def get_payoffs_vector(self) -> List[int]:
        """
        :return: The payoff vector as provided at initialization
        """
        return self.vector
