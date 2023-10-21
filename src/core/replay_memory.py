import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import Tuple

import numpy as np

Transition = namedtuple("Transition", field_names=["state", "action", "reward", "done", "new_state"])


class AbstractReplayMemory(ABC):
    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def push(self, transition: Transition):
        """Save a transition"""
        ...

    @abstractmethod
    def sample(self, batch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # choose batch_size transitions from memory, based on some strategy
        ...


class ReplayMemory(AbstractReplayMemory):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition: Transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = min(batch_size, len(self.memory))
        # choose batch_size transitions from memory, randomly picked
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        # Get "columns" of states, actions, etc. as new arrays using zip and unpacking
        states, actions, rewards, dones, next_states = zip(*[self.memory[idx] for idx in indices])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class PrioritizedReplayMemory(AbstractReplayMemory):
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment_per_sampling=0.001,
                 epsilon: float = 1e-6):
        """
        :param capacity: size of replay buffer
        :param alpha: scaling constant for computing probabilities out of priorities
        :param beta: scaling constant for importance weights
        :param epsilon: offset to avoid zero probabilities
        """
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon

    def __len__(self):
        return len(self.memory)

    def push(self, transition: Transition):
        """Save a transition"""
        self.memory.append(transition)

        # new experience -> the highest priority
        self.priorities.append(max(self.priorities, default=1))

    def update(self, idx, error):
        p = self._get_priority(error)
        self.priorities[idx] = p

    def get_probabilities(self):
        """
        :return: probabilities of sampling each transition
        """
        scaled_priorities = np.array(self.priorities) ** self.alpha
        return scaled_priorities / sum(scaled_priorities)

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def get_importance(self, probabilities):
        """
        :param probabilities: probabilities of sampling each transition
        :return: importance weights
        """
        importance = 1 / len(self.memory) * 1 / probabilities
        importance = importance ** self.beta
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size):
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        batch_size = min(batch_size, len(self.memory))

        sample_probabilities = self.get_probabilities()
        sample_indices = random.choices(range(len(self.memory)), weights=sample_probabilities, k=batch_size)

        states, actions, rewards, dones, next_states = zip(*[self.memory[idx] for idx in sample_indices])
        importances = self.get_importance(sample_probabilities[sample_indices])
        env_data = (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
            np.array(sample_indices),
            np.array(importances),
        )
        return env_data
