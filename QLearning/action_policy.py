from typing import List
import numpy as np

class ActionPolicy:
    def select(self, q: List[float]) -> int:
        raise NotImplementedError
    
class EpsilonGreedy(ActionPolicy):
    epsilon: float
    
    def __init__(self, epsilon) -> None:
        super().__init__()
        self.epsilon = epsilon
    
    def select(self, q: List[float]) -> int:
        be_greedy = np.random.uniform() > self.epsilon
        if be_greedy:
            argmax = np.argmax(q)
            return argmax
        else:
            choice = np.random.choice(len(q))
            return choice