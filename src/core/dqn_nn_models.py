import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP_DQN(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size, n_actions: int, hidden_size: int = 128, num_hidden_layers: int = 1):
        super(MLP_DQN, self).__init__()

        input_layer = nn.Linear(math.prod(obs_size), hidden_size)
        hidden_layers = [input_layer]
        for _ in range(num_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())
        output_layer = nn.Linear(hidden_size, n_actions)
        hidden_layers.append(output_layer)

        self.net = nn.Sequential(
            *hidden_layers
        )

    def forward(self, x):
        batch_size = x.shape[0]
        flattened_x = x.view(batch_size, -1)

        return self.net(flattened_x.float())


class CNN_DQN(nn.Module):
    """
    CNN network
    Args:
        obs_size: observation/state size of the environment (2D input)
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers (flattened)
    """

    def __init__(self, obs_size, n_actions, hidden_size=32, num_hidden_layers: int = 1):
        super().__init__()

        height, width, channels = obs_size

        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
        )

        first_fcl = nn.Linear(self.get_flattened_output_shape(obs_size), hidden_size)
        hidden_layers = [first_fcl]
        for _ in range(num_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())
        output_layer = nn.Linear(hidden_size, n_actions)
        hidden_layers.append(output_layer)

        self.mlp = nn.Sequential(
            *hidden_layers
        )

    @torch.no_grad()
    def get_flattened_output_shape(self, obs_size):
        dummy_input = torch.zeros(1, *obs_size)
        dummy_input = dummy_input.permute(0, 3, 1, 2)
        output = self.conv_net(dummy_input)
        return output.shape[1]

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        flattened_conv_output = self.conv_net(x.float())
        mlp_output = self.mlp(flattened_conv_output)

        return mlp_output


class DuelingDQNNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        flattened_x = x.view(batch_size, -1).float()

        feature_x = self.feature(flattened_x)
        advantage = self.advantage(feature_x)
        value = self.value(feature_x)
        return value + advantage - advantage.mean()


class MLP_DQN_NOISY(nn.Module):
    """
    MLP network with noisy layers. Weights are now defined by means / stds, which are learnable stochastic parameters
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size, n_actions: int, hidden_size: int = 128, num_hidden_layers: int = 1):
        super(MLP_DQN_NOISY, self).__init__()

        input_layer = NoisyLinear(math.prod(obs_size), hidden_size)
        hidden_layers = [input_layer]
        for _ in range(num_hidden_layers):
            hidden_layers.append(NoisyLinear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())
        output_layer = NoisyLinear(hidden_size, n_actions)
        hidden_layers.append(output_layer)

        self.net = nn.Sequential(
            *hidden_layers
        )

    def forward(self, x):
        batch_size = x.shape[0]
        flattened_x = x.view(batch_size, -1)

        return self.net(flattened_x.float())


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # weights have shape out x in so the calculation "input @ weight.T + bias" works
        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_std = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mean = nn.Parameter(torch.empty(out_features))
        self.bias_std = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # normalize mean range by input features
        mean_range = 1 / math.sqrt(self.weight_mean.size(1))

        self.weight_mean.data.uniform_(-mean_range, mean_range)
        self.weight_std.data.fill_(self.std_init / math.sqrt(self.weight_std.size(1)))

        self.bias_mean.data.uniform_(-mean_range, mean_range)
        self.bias_std.data.fill_(self.std_init / math.sqrt(self.bias_std.size(0)))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon = epsilon_out.outer(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mean + self.weight_std * self.weight_epsilon,
                            self.bias_mean + self.bias_std * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mean, self.bias_mean)
