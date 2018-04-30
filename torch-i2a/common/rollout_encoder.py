import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd



class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, num_rewards, hidden_size):
        super(RolloutEncoder, self).__init__()

        self.in_shape = in_shape

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.gru = nn.GRU(self.feature_size() + num_rewards, hidden_size)

    def forward(self, state, reward):
        num_steps  = state.size(0)
        batch_size = state.size(1)

        # In shape is just the shape of the state space
        state = state.view(-1, *self.in_shape)
        # Extract features from the state space using a conv net
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)

        # Process sequence of frames with RNN to return final score for
        # sequence
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)


