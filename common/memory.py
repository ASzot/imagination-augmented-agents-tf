import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward',
    'next_state', 'done'))


class EpisodicReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.memory.append([])  # List for first episode
    self.position = 0

  def append(self, state, action, reward, next_state, done):
    self.memory[self.position].append(Transition(state, action, reward,
        next_state, done))
    # Terminal states are saved with actions as None, so switch to next episode
    if action is None:
      self.memory.append([])
      self.position = min(self.position + 1, self.num_episodes - 1)

  # Samples random trajectory
  def sample(self, maxlen=0):
    while True:
      e = random.randrange(len(self.memory))
      mem = self.memory[e]
      T = len(mem)
      if T > 0:
        # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
        if maxlen > 0 and T > maxlen + 1:
          t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
          return mem[t:t + maxlen + 1]
        else:
          return mem

  # Samples batch of trajectories, truncating them to the same length
  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    minimum_size = min(len(trajectory) for trajectory in batch)
    batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def __len__(self):
    return sum(len(episode) for episode in self.memory)



class RolloutStorage(object):
    def __init__(self, num_steps, num_envs, state_shape):
        self.num_steps = num_steps
        self.num_envs  = num_envs

        states_shape = [0] * (len(state_shape) + 2)
        states_shape[0] = num_steps + 1
        states_shape[1] = num_envs
        for i in range(len(state_shape)):
            states_shape[i + 2] = state_shape[i]
        self.states  = np.zeros(states_shape)
        self.rewards = np.zeros((num_steps,     num_envs, 1))
        self.masks   = np.ones((num_steps  + 1, num_envs, 1))
        self.actions = np.zeros((num_steps,     num_envs, 1))
        self.use_cuda = False


    def insert(self, step, state, action, reward, mask):
        self.states[step + 1] = np.copy(state)
        self.actions[step] = np.copy(action)
        self.rewards[step] = np.copy(reward)
        self.masks[step + 1] = np.copy(mask)


    def after_update(self):
        self.states[0] = np.copy(self.states[-1])
        self.masks[0] = np.copy(self.masks[-1])


    def compute_returns(self, next_value, gamma):
        returns   = np.zeros((self.num_steps + 1, self.num_envs, 1))
        returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return returns[:-1]


