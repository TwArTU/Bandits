from typing import List
import numpy as np
from learning_rule import LearningRule
from action_policy import ActionPolicy
from copy import deepcopy

class BanditsAgent():
    '''
    LSTM の場合は入力にr_{t-1} を使うが、Q-Learning の場合は不要
    逆にo_{t-1} が必要
    '''
    q_table: List[List[float]]
    learning_rule: LearningRule
    action_policy: ActionPolicy
    pre_action: int
    pre_reward: int
    
    def __init__(self, m: int, n: int, learning_rule: LearningRule, action_policy: ActionPolicy, q_uniformed=False, pre_state=None, pre_action=None, pre_reward=None) -> None:
        if q_uniformed: # 小さな値で初期化するか、ゼロで初期化するか
            self.q_table = np.random.uniform(size=(m,n))
        else:
            self.q_table = np.zeros((m,n))
        self.learning_rule = learning_rule
        self.action_policy = action_policy
        self.pre_action = pre_action
        self.pre_reward = pre_reward
        self.pre_state = pre_state
    
    def select_arm(self, state):
        choice = self.action_policy.select(self.q_table[state])
        return choice

    def update_qtable(self, state, reward):
        self.learning_rule.update(self.q_table, state, self.pre_state, self.pre_action, reward)
    
    def run(self, state, env, step):
        action = self.select_arm(state)
        reward = env.get_reward(state, action, step)
        if self.pre_state is None or self.pre_action is not None and self.pre_reward is not None:
            self.update_qtable(state, reward)
        self.pre_state = state
        self.pre_action = action
        self.pre_reward = reward
        q_copy = deepcopy(self.q_table)
        return q_copy, action, reward