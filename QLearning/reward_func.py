import numpy as np

class RewardFunc():
    def get_reward(self, state: int, action: int, step: int) -> float:
        raise NotImplementedError
    
class Reward4TwoArmed(RewardFunc):
    '''
    m-machines, 2 armed bandits 用のReward
    reward probability はランダム
    '''
    def __init__(self, m: int) -> None:
        super().__init__()
        self.reward_prob = []
        for _ in range(m):
            p1 = np.random.rand()
            p2 = 1 - p1
            self.reward_prob.append([p1,p2])
            
    def get_reward(self, state: int, action: int, step: int) -> float:
        rand = np.random.uniform()
        if rand < self.reward_prob[state][action]:
            reward = 1
        else:
            reward = 0
        return reward
        
class Reward4TwoArmed50(Reward4TwoArmed):
    '''
    reward probability p_0 = p_1 = 0.5
    '''
    def __init__(self, m) -> None:
        self.reward_prob = np.array([[0.5,0.5] for _ in range(m)])
        
class Reward4TwoArmed75(Reward4TwoArmed):
    '''
    reward probability p_0 = 0.25, p_1 = 0.75
    '''
    def __init__(self, m) -> None:
        self.reward_prob = np.array([[0.25,0.75] for _ in range(m)])