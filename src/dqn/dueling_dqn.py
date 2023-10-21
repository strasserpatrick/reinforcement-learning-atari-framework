import math

import torch

from core import DuelingDQNNetwork
from dqn.abstract_dqn import AbstractDQN
from utils.config import DQNConfig


class DuelingDQN(AbstractDQN):
    def __init__(self, hparams: DQNConfig):
        super().__init__(hparams)

        self.online_net = DuelingDQNNetwork(math.prod(self.observation_size), self.action_space)
        self.target_net = DuelingDQNNetwork(math.prod(self.observation_size), self.action_space)

        # initialize target network with mirrored weights
        self.target_net.load_state_dict(self.online_net.state_dict())

    def compute_expected_q_values(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            # Take the q value from the best action determined by the target net
            next_state_values = self.target_net(next_states).max(axis=1)[
                0]  # index 0 returns values from a torch tensor
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_q_values = rewards + self.hparams.gamma * next_state_values
        return expected_q_values
