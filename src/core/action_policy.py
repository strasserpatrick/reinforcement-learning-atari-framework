import math
import random
from abc import ABC, abstractmethod

import torch
from torch import nn


class ActionPolicy(ABC):
    """Abstract class for action policy."""

    @abstractmethod
    def select_action(self, state, env, model):
        """Get action from state."""
        ...

    @abstractmethod
    def return_random_action(self, env):
        """Return random action."""
        ...

    @abstractmethod
    def return_best_action(self, state, model):
        """Return best action."""
        ...


class EpsilonGreedyPolicy(ActionPolicy):
    """Epsilon greedy action policy."""

    def __init__(self, device, eps_start=0.9, eps_end=0.05, eps_decay=1000):
        super().__init__()

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.device = device

        self.steps_done = 0

    def select_action(self, state, env, model):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.return_best_action(state, model)
        else:
            return self.return_random_action(env)

    def return_random_action(self, env):
        return env.action_space.sample()

    @torch.no_grad()
    def return_best_action(self, state, model: nn.Module) -> int:
        state = torch.from_numpy(state).to(self.device)
        state_batched = state.unsqueeze(0)
        best_action = model.forward(state_batched).max(1)[1].view(1, 1).item()
        return int(best_action)
