from typing import List, Tuple


# Runner needs this

class Matrix_Payoffs:
    def __init__(self, payoffs):
        self.num_agents = 2
        self.vector = payoffs

    def get_payoff(self, actions: List[int]) -> Tuple[int]:
        pass

    def get_payoffs_vector(self) -> List[int]:
        return self.vector
