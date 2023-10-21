from typing import Tuple

import gymnasium as gym
import torch
from torch import nn

from core.action_policy import EpsilonGreedyPolicy
from core.replay_memory import Transition, AbstractReplayMemory


class Agent:
    """
    Base Agent class handling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(
            self,
            env: gym.Env,
            eps_start: float,
            eps_end: float,
            eps_decay: float,
            device: torch.device,
            replay_buffer: AbstractReplayMemory = None,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state, _ = self.env.reset()

        self.device = device
        self.action_policy = EpsilonGreedyPolicy(
            device=self.device,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
        )

    def reset(self) -> None:
        """Resents the environment and updates the state"""
        self.state, _ = self.env.reset()

    @torch.no_grad()
    def play_step(self, net: nn.Module, random_action=False, best_action=False) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment
        Args:
            net: DQN network
            random_action: whether to take a random action instead of the selected action by policy
            best_action: whether to choose the best possible action
        Returns:
            reward, done
        """

        if random_action:
            action = self.action_policy.return_random_action(self.env)
        elif best_action:
            action = self.action_policy.return_best_action(self.state, net)
        else:
            action = self.action_policy.select_action(self.state, self.env, net)

        # do step in the environment
        new_state, reward, terminated, truncated, __ = self.env.step(action)
        done = terminated or truncated
        reward = float(reward)

        transition = Transition(state=self.state, action=action, reward=reward, done=done, new_state=new_state)

        if self.replay_buffer is not None:
            self.replay_buffer.push(transition)

        self.state = new_state
        if done:
            self.reset()

        return reward, done
