# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:42:27 2021

@author: andy
"""

import gym
from typing import TypeVar
import  random
from save_gif import save_frames_as_gif

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.8):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))

    obs = env.reset()
    total_reward = 0.0

    frames = []
    while True:
        frames.append(env.render(mode="rgb_array"))
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            env.close()
            save_frames_as_gif(frames, filename='random_action_wrapper.gif')
            break

    print("Reward got: %.2f" % total_reward)