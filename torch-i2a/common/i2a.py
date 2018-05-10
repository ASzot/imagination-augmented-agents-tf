import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from common.actor_critic import OnPolicy
from common.rollout_encoder import RolloutEncoder

from common.torch_util import Variable

class I2A(OnPolicy):
    def __init__(self, in_shape, num_actions, num_rewards, hidden_size, imagination, full_rollout=True):
        super().__init__()

        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards

        self.imagination = imagination

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)

        if full_rollout:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + num_actions * hidden_size, 256),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + hidden_size, 256),
                nn.ReLU(),
            )

        self.critic  = nn.Linear(256, 1)
        self.actor   = nn.Linear(256, num_actions)

    def forward(self, state):
        # Batch size is first element of the state input
        # This will be the number of environments multiplied by the action
        # space
        batch_size = state.size(0)

        # Get a full rollout of an imagined sequence
        # This will be a tensor of size
        # [rollout count, # envs (batch size) * actions, *state space]
        imagined_state, imagined_reward = self.imagination(state.data)
        self.imagined_state = imagined_state
        self.imagined_reward = imagined_reward

        hidden = self.encoder(Variable(imagined_state), Variable(imagined_reward))
        # Get encoded representation of each state
        hidden = hidden.view(batch_size, -1)

        self.encoded_repr = hidden

        # Extract features from state
        state = self.features(state)
        state = state.view(state.size(0), -1)

        # Input is a weighted sum of imagination scores and extracted features
        # from input. The state is the model free path and the hidden
        # is the model based path
        x = torch.cat([state, hidden], 1)
        x = self.fc(x)

        # Use our standard policy from a2c
        logit = self.actor(x)
        value = self.critic(x)

        return logit, value

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)


