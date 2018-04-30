import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from common.torch_util import Variable

from common.pacman_util import target_to_pix


# The output of this is
# [rollout count, # envs (batch size) * actions, *state space]
class ImaginationCore(object):
    def __init__(self, num_rolouts, in_shape, num_actions, num_rewards, env_model, distil_policy, full_rollout=True):
        self.num_rolouts  = num_rolouts
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        self.env_model     = torch.nn.DataParallel(env_model, device_ids=[0,1,2]).cuda()
        self.distil_policy = distil_policy
        self.full_rollout  = full_rollout

    def __call__(self, state):
        state      = state.cpu()
        batch_size = state.size(0)

        rollout_states  = []
        rollout_rewards = []

        if self.full_rollout:
            state = state.unsqueeze(0).repeat(self.num_actions, 1, 1, 1, 1).view(-1, *self.in_shape)
            action = torch.LongTensor([[i] for i in range(self.num_actions)] * batch_size)
            action = action.view(-1)
            rollout_batch_size = batch_size * self.num_actions
        else:
            print('NOT USING FULL ROLLOUT')
            action = self.distil_policy.act(Variable(state, volatile=True))
            action = action.data.cpu()
            rollout_batch_size = batch_size
            raise ValueError('CANNOT USE FULL ROLLOUT')

        for step in range(self.num_rolouts):
            # Creating the whole thing to be ones to start off with assumes
            # that
            # batch size 80
            # [400, 5, 15, 19]

            # Encode the actions
            onehot_action = torch.zeros(rollout_batch_size, self.num_actions, *self.in_shape[1:])
            onehot_action[range(rollout_batch_size), action] = 1

            #if not (np.all(x == 1)):
            #    raise ValueError('NOT ALL EQUAL ONE')

            # Combination of the pixel frames and the actions are the input to
            # the environment model. Note that for a full roll out we are
            # taking every single action and then evaluating it
            inputs = torch.cat([state, onehot_action], 1)

            # Imagine next states and rewards
            imagined_state, imagined_reward = self.env_model(Variable(inputs, volatile=True))

            imagined_state  = F.softmax(imagined_state, dim=1).max(1)[1].data.cpu()
            imagined_reward = F.softmax(imagined_reward, dim=1).max(1)[1].data.cpu()

            imagined_state = target_to_pix(imagined_state.numpy())
            imagined_state = torch.FloatTensor(imagined_state).view(rollout_batch_size, *self.in_shape)

            onehot_reward = torch.zeros(rollout_batch_size, self.num_rewards)
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state.unsqueeze(0))
            rollout_rewards.append(onehot_reward.unsqueeze(0))

            state  = imagined_state
            action = self.distil_policy.act(Variable(state, volatile=True))
            action = action.data.cpu()

        return torch.cat(rollout_states), torch.cat(rollout_rewards)


