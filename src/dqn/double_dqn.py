import torch

from dqn.abstract_dqn import AbstractDQN
from utils.config import DQNConfig


class DoubleDQN(AbstractDQN):
    def __init__(self, hparams: DQNConfig):
        super().__init__(hparams)

        self.online_net = self.create_net()
        self.target_net = self.create_net()

        # initialize target network with mirrored weights
        self.target_net.load_state_dict(self.online_net.state_dict())

    def compute_expected_q_values(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            selected_action_indices = torch.argmax(self.online_net(next_states), dim=1)
            next_q_values = self.target_net(next_states)
            next_state_values = next_q_values[torch.arange(next_q_values.size(0)), selected_action_indices]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_q_values = rewards + self.hparams.gamma * next_state_values
        return expected_q_values
