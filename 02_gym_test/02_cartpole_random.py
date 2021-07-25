# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:41:41 2021

@author: andy
"""

import gym
from save_gif import save_frames_as_gif


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    frames = []
    while True:
        frames.append(env.render(mode="rgb_array"))
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            env.close()
            save_frames_as_gif(frames, filename='cartpole_random.gif')
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))