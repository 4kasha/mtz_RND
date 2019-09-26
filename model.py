# Actor-Critic & RND network
# ref. https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCritic(nn.Module):
    def __init__(self, input_size, output_size, seed):
        super(CnnActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(6 * 6 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 448),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(448, 448),
            nn.ReLU(),
            nn.Linear(448, output_size)
        )

        self.extra_layer = nn.Sequential(
            nn.Linear(448, 448),
            nn.ReLU()
        )

        self.critic_ext = nn.Linear(448, 1)
        self.critic_int = nn.Linear(448, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        action_scores = self.actor(x)
        action_probs = F.softmax(action_scores, dim=1)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        return action_probs, value_ext, value_int


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size, seed):
        super(RNDModel, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        feature_output = 7 * 7 * 64

        self.target = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # Initialize weights    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
