# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:00:46 2021

@author: andy
"""

import collections
from random import sample

GAMMA = 0.9
TEST_EPISODES = 20


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
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        self.env.reset()
        for _ in range(count):
            action = sample(self.env.action_space, 1)[0]
            new_state, reward, is_done = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
            

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in self.env.action_space:
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self):
        total_reward = 0.0
        state = self.env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done = self.env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in self.env.all_state:
            state_values = [self.calc_action_value(state, action)
                            for action in self.env.all_state[state]]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    agent = Agent()

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        print(agent.values)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode()
        reward /= TEST_EPISODES
        print(reward)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 3:
            print("Solved in %d iterations!" % iter_no)
            break