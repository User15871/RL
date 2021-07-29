# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:00:46 2021

@author: andy
"""

import collections
from random import sample

GAMMA = 0.9
TEST_EPISODES = 20
ALPHA = 0.2


class Env:
    def __init__(self):
        self.all_state = {'1':{'r':'2', 'd':'3'}, '3':{'r':'7','d':'4'},'4':{'d':'5'},'5':{'d':'6'},'7':{'r':'8','d':'9'}}
        self.all_score = {'1':{'r':1, 'd':2}, '3':{'r':1,'d':2},'4':{'d':2},'5':{'d':-20},'7':{'r':-20,'d':1}}
        self.state = '1'
        self.action_space = [x for x in self.all_state[self.state]]
    
    def reset(self):
        self.state = '1'
        self.action_space = [x for x in self.all_state[self.state]]
        return self.state
    
    def step(self, action):
        score = self.all_score[self.state][action]
        self.state = self.all_state[self.state][action]   
        if self.state in self.all_state:
            self.action_space = [x for x in self.all_state[self.state]]
        else:
            self.action_space = []
        return self.state, score, (len(self.action_space) == 0)

class Agent:
    def __init__(self):
        self.env = Env()
        self.values = collections.defaultdict(float)
        self.i = 0
        
    def sample_env(self):
        state = self.env.state
        if self.i % 2 == 0:
            action = sample(self.env.action_space, 1)[0]
        else:
            _, action = self.best_value_and_action(state)
        self.i += 1
        new_state, reward, is_done = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (state, action, reward, new_state)

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in self.env.action_space:
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA

    def play_episode(self):
        total_reward = 0.0
        state = self.env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done = self.env.step(action)
            total_reward += reward
            if is_done:
                self.env.reset()
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    agent = Agent()

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        for i in range(10):
            s, a, r, next_s = agent.sample_env()
            agent.value_update(s, a, r, next_s)
        print(agent.values)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode()
        reward /= TEST_EPISODES
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 3:
            print("Solved in %d iterations!" % iter_no)
            break