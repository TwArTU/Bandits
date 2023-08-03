from reward_func import RewardFunc

class BanditsEnv():
    m: int
    n: int
    reward_func: RewardFunc
    
    def __init__(self, m: int, n: int, reward_func: RewardFunc) -> None:
        self.m = m
        self.n = n
        self.reward_func = reward_func
    
    def get_reward(self, state, action, step):
        return self.reward_func.get_reward(state, action, step)