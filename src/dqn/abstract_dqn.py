from abc import ABC, abstractmethod
from typing import Tuple, List
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from core import Agent, ReplayMemory, PrioritizedReplayMemory, RLDataset
from core import MLP_DQN, CNN_DQN
from core.dqn_nn_models import MLP_DQN_NOISY, NoisyLinear
from utils.config import DQNConfig, Networks


class AbstractDQN(pl.LightningModule, ABC):
    def __init__(self, hparams: DQNConfig):
        super().__init__()
        self.hparams.update(vars(hparams))  # https://github.com/Lightning-AI/lightning/discussions/7525

        self.my_device = torch.device(self.hparams.device)
        self.env = gym.make(self.hparams.env)
        self.observation_size = self.env.observation_space.shape
        self.action_space = self.env.action_space.n

        if self.hparams.replay_buffer_type == "vanilla":
            self.buffer = ReplayMemory(self.hparams.replay_size)
        elif self.hparams.replay_buffer_type == "prioritized":
            self.buffer = PrioritizedReplayMemory(self.hparams.replay_size)
        else:
            raise ValueError(f"Unknown replay buffer type {self.hparams.replay_buffer_type}")

        self.loss_fn = nn.MSELoss(reduce=False)

        self.online_net = None

        self.episode_reward = 0
        self.total_reward = 0

        self.agent = Agent(
            env=self.env,
            replay_buffer=self.buffer,
            eps_start=self.hparams.eps_start,
            eps_end=self.hparams.eps_end,
            eps_decay=self.hparams.eps_decay,
            device=self.my_device
        )

        # Initially fill the replay buffer
        self.populate(self.hparams.warm_start_steps)
        self.save_hyperparameters()

    @staticmethod
    def _unpack_batch_essentials(batch):
        states = batch[0]
        actions = batch[1]
        rewards = batch[2]
        dones = batch[3]
        next_states = batch[4]

        return states, actions, rewards, dones, next_states

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        states, actions, rewards, dones, next_states = self._unpack_batch_essentials(batch)

        # Best action index for each state by online net
        q_values = self.compute_q_values(states=states, actions=actions)
        expected_q_values = self.compute_expected_q_values(states, actions, rewards, next_states, dones)

        if self.hparams.replay_buffer_type == "prioritized":
            indices = batch[5]
            importances = batch[6]

            self._update_priority_queue(indices, q_values=q_values, expected_q_values=expected_q_values)
            loss = self._compute_priority_queue_loss(q_values, expected_q_values, importances)

        else:  # this is always the vanilla case
            loss = self.loss_fn(q_values, expected_q_values).mean()

        return loss

    def _update_priority_queue(self, indices, q_values, expected_q_values) -> None:
        """
        Update the priority queue with the new errors
        :param indices: which entries to update
        :param q_values: predicted q values for error computation
        :param expected_q_values: expected q values for error computation
        :return: None
        """
        errors = torch.abs(q_values - expected_q_values).detach().cpu().numpy()
        for i in range(len(indices)):
            idx = indices[i]
            self.buffer.update(idx, errors[i])

    def _compute_priority_queue_loss(self, q_values, expected_q_values, importances):
        """
        Compute the loss for the prioritized replay buffer
        :param q_values: predicted q values for error computation
        :param expected_q_values: expected q values for error computation
        :param importances: weight balancing factors for prioritized replay buffer
        :return: loss
        """
        return (self.loss_fn(q_values, expected_q_values) * importances).mean()

    def compute_q_values(self, states, actions):
        """
        Calculates the q values for the current mini batch of replay data
        :param states:
        :param actions:
        :return:
        """
        q_values = self.online_net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        return q_values

    @abstractmethod
    def compute_expected_q_values(self, states, actions, rewards, next_states, dones):
        """
        Calculates the expected q values for the current mini batch of replay data
        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param dones:
        :return: expected_q_values
        """
        ...

    def training_step(
            self,
            batch,
            nb_batch,
    ) -> OrderedDict:
        device = self.get_device(batch)

        # normally, don't choose the best action but explore using epsilon greedy
        # when using noisy dqn, choose action with highest q value as exploration is handled by noise in layers
        choose_best_action = False
        if self.hparams.network_type == "mlp_noisy":
            # when using noisy dqn, choose action with highest q value as exploration is handled by noise in layers
            choose_best_action = True
        reward, done = self.agent.play_step(self.online_net, best_action=choose_best_action)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)
        self.total_reward += reward

        log = {
            "reward": torch.tensor(reward).to(device).item(),
            "total_reward": torch.tensor(self.total_reward).to(device).item(),
        }

        if done:
            self.log("episode_reward", self.episode_reward)
            log["episode_reward"] = self.episode_reward

            self.episode_reward = 0

        # Soft update of target network every sync_rate steps
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        if self.global_step % self.hparams.validate_every_n_epochs == 0:
            validation_episode_reward = self.validate()
            # CSV Logger does not support on_step logging!, on_epoch needs to be forced here
            self.log("validation_reward", validation_episode_reward, on_epoch=True)
            log["validation_reward"] = validation_episode_reward

        self.log("reward", reward)
        self.log("total_reward", self.total_reward)
        self.log("loss", loss)

        # reset noise after every update when using noisy layers
        if self.hparams.network_type == "mlp_noisy":
            for layer in self.online_net.modules():
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        optimizer = optim.Adam(self.online_net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def validate(self) -> float:
        """
        Validates the current network by playing an episode without exploration
        :return: episode reward of validation run
        """

        validation_env = gym.make(self.hparams.env)

        validation_agent = Agent(
            env=validation_env,
            eps_start=self.hparams.eps_start,
            eps_end=self.hparams.eps_end,
            eps_decay=self.hparams.eps_decay,
            device=self.hparams.device
        )

        validation_rollout_rewards = []

        for _ in range(self.hparams.rollouts_per_validation):
            validation_rollout_reward = 0
            while True:
                reward, done = validation_agent.play_step(self.online_net, best_action=True)
                validation_rollout_reward += reward
                if done:
                    break
            validation_rollout_rewards.append(validation_rollout_reward)

        return float(np.median(np.array(validation_rollout_rewards)))

    def create_net(self):
        """
        creates the network based on argument description.
        This is either a MLP or CNN network

        :return: nn.Module
        """

        if self.hparams.network_type == 'mlp':
            return MLP_DQN(self.observation_size, self.action_space, self.hparams.hidden_size,
                           self.hparams.num_hidden_layers)
        elif self.hparams.network_type == 'mlp_noisy':
            return MLP_DQN_NOISY(self.observation_size, self.action_space, self.hparams.hidden_size,
                                 self.hparams.num_hidden_layers)
        elif self.hparams.network_type == 'cnn':
            return CNN_DQN(self.env.observation_space.shape, self.action_space, self.hparams.hidden_size)
        else:
            raise ValueError('Network type not supported')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        return self.online_net(x)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.online_net, random_action=True)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else "cpu"

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader
