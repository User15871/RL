# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 19:32:03 2021

@author: andy
"""

from lib import wrappers
from lib import dqn_model

import  argparse
import  time
import numpy as np
import collections
from save_gif import Gif

import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.0
STEP_COUNT = 2

PRIOR_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.max_prio = 1.0
        self.prob_alpha = prob_alpha

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, i):
        return self.buffer[i]
    
    
    def __reversed__(self):
        return reversed(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
        self.priorities.append(self.max_prio)

    def sample(self, batch_size, beta = 0.4):
        probs = np.array(self.priorities, dtype=np.float32) ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** beta
        
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states), indices, weights
               
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
        self.max_prio = batch_priorities.max()
    
    def clear(self):
        self.buffer.clear()
        

class  Agent :
    def __init__(self, env, exp_buffer, steps_count = 1, gamma = 0.99):
        
        self.env = env
        self.gamma = gamma
        self.hist_buffer = ExperienceBuffer(steps_count)
        self.exp_buffer = exp_buffer
        self.gif = Gif(path = './pong/')
        self._reset()

    def _reset(self):
        self.total_reward = 0.0
        self.state = env.reset()
        self.hist_buffer.clear()

    
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        self.gif.load_frame(self.env.render(mode="rgb_array"))
        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
            
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.hist_buffer.append(exp)
        self.state = new_state
        
        hist_reward = 0.0
        for e in reversed(self.hist_buffer):
            hist_reward *= self.gamma
            hist_reward += e.reward
            
        self.exp_buffer.append(Experience(self.hist_buffer[0].state, self.hist_buffer[0].action, hist_reward, self.hist_buffer[-1].done, self.hist_buffer[-1].new_state))
            
        if is_done:
            for end_buffer_step in range(1, len(self.hist_buffer)):
                hist_reward = 0.0
                for i, e in enumerate(reversed(self.hist_buffer)):
                    if i == end_buffer_step:
                        break
                    hist_reward *= self.gamma
                    hist_reward += e.reward
                    
                self.exp_buffer.append(Experience(self.hist_buffer[end_buffer_step].state, self.hist_buffer[end_buffer_step].action, hist_reward, self.hist_buffer[-1].done, self.hist_buffer[-1].new_state))
                done_reward = self.total_reward
                self.gif.save_gif()
            self._reset()
        return done_reward


def calc_loss(states, actions, rewards, dones, next_states, weights, net, tgt_net, device="cpu", double=True):
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    batch_weights_v = torch.tensor(weights).to(device)
    state_action_values = net(states_v).gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    
    expected_state_action_values = next_state_values.detach() * (GAMMA ** STEP_COUNT) + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
#    print("rewards_v:", rewards_v)
#    print("expected_state_action_values:", expected_state_action_values)
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    net = dqn_model.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer, STEP_COUNT)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START)/BETA_FRAMES)
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            mean_reward = np.mean(total_rewards[-100:])
            ts_frame = frame_idx
            ts = time.time()
            
            print("%d: done %d games, mean reward %.3f, reward %.3f, eps %.2f, speed %.2f f/s, max_loss %.2f" % (
                frame_idx, len(total_rewards), mean_reward, reward, epsilon,
                speed, buffer.max_prio
            ))
            
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        states, actions, rewards, dones, next_states, batch_indices, weights  = buffer.sample(BATCH_SIZE, beta)
        loss_t, sample_prios = calc_loss(states, actions, rewards, dones, next_states, weights, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
    writer.close()