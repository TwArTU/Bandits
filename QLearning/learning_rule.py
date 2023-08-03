from typing import List

class LearningRule:
    def update(self, q: List[List[float]], new_reward: float, new_state: int, state: int, action: int) -> None:
        raise NotImplementedError
    
class QLearning(LearningRule):
    alpha: float    # 学習率
    gamma: float    # 割引率

    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def update(self, q: List[List[float]], state: int, pre_state: int, pre_action: int, reward: float) -> None:
        '''
        Q(s,a) ← Q(s,a) + α ･ (r + γ ･ max_{a'} Q(s', a') - Q(s,a))
        '''
        q[pre_state][pre_action] = q[pre_state][pre_action] + self.alpha * (reward + self.gamma*max(q[state]) - q[pre_state][pre_action])
