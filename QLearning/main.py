from typing import List
import numpy as np
import matplotlib.pyplot as plt
from bandits_agent import BanditsAgent
from bandits_env import BanditsEnv
from action_policy import EpsilonGreedy
from learning_rule import QLearning
from reward_func import RewardFunc, Reward4TwoArmed75, Reward4TwoArmed50, Reward4TwoArmed
from copy import deepcopy

class Bandits():
    '''
    agent から出力されるresult をまとめる
    シミュレーションの流れはBanditsAgent のrun を参照
    '''
    agent: BanditsAgent
    env: BanditsEnv
    
    def __init__(self, agent: BanditsAgent, env: BanditsEnv) -> None:
        self.agent = agent
        self.env = env
        self.steps = 0
        
    def run(self, states: List[int]):
        '''
        states を入力として受け取る
        '''
        q_res, a_res, r_res = [], [], []
        for state in states:
            self.steps += 1
            q,a,r = self.agent.run(state, self.env, self.steps)
            q_res.append(deepcopy(q))
            a_res.append(a)
            r_res.append(r)
        return q_res, a_res, r_res

def generate_inputs(m: int, maxstep: int) -> List[int]:
    return np.random.choice(m, maxstep)

if __name__=='__main__':
    # SEED = 123456789
    # np.random.seed(SEED)
    m = 1
    n = 2
    epsilon = 0.1
    alpha = 0.007
    gamma = 0.9
    training_steps = 20000      # トレーニング回数
    training_inputs = generate_inputs(m, training_steps)
    posttraining_steps = 200    # トレーニング後の学習結果を見る
    posttraining_inputs = generate_inputs(m, posttraining_steps)

    def reward_sim(reward_func: RewardFunc, figname: str):
        '''
        reward function 比較用
        実行部分ではreward probability が異なり、reward function は同じものを比較している
        '''
        action_policy = EpsilonGreedy(epsilon)
        learning_rule = QLearning(alpha, gamma)
        agent = BanditsAgent(m,n,learning_rule,action_policy)
        env = BanditsEnv(m,n,reward_func)
        bandits = Bandits(agent, env)
        
        _training_results = bandits.run(training_inputs)
        
        posttraining_results = bandits.run(posttraining_inputs)
        
        q_table_hist, action_hist, reward_hist = posttraining_results
        # print(len(q), q)
        # print(a)
        # print(r)
        plt.plot(np.arange(posttraining_steps), reward_hist)
        plt.scatter(np.arange(posttraining_steps), action_hist, s=10, c='magenta')
        
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.savefig(figname, dpi=300)
        plt.cla()
        
    # if 0:
    #     rf = Reward4TwoArmed(m)
    #     fn = "Reward4TwoArmed"
    #     sim_reward(rf, fn)
    if 1:
        '''
        reward probability p_0 = p_1 = 0.5
        '''
        rf = Reward4TwoArmed50(m)
        fn = "Reward4TwoArmed50"
        reward_sim(rf, fn)
    if 1:
        '''
        reward probability p_0 = 0.25, p_1 = 0.75
        '''
        rf = Reward4TwoArmed75(m)
        fn = "Reward4TwoArmed75"
        reward_sim(rf, fn)