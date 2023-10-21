from utils.config import DQNType, DQNConfig
from .abstract_dqn import AbstractDQN
from .double_dqn import DoubleDQN
from .rainbow_dqn import RainbowDQN
from .regular_dqn import RegularDQN
from .dueling_dqn import DuelingDQN


def dqn_model_finder(dqn_type: DQNType):
    if dqn_type == DQNType.RegularDQN:
        return RegularDQN
    elif dqn_type == DQNType.DuelingDQN:
        return DuelingDQN
    elif dqn_type == DQNType.DoubleDQN:
        return DoubleDQN
    elif dqn_type == DQNType.RainbowDQN:
        return RainbowDQN
    else:
        raise NotImplementedError


def dqn_factory(dqn_type: DQNType, hparams: DQNConfig) -> AbstractDQN:
    return dqn_model_finder(dqn_type)(hparams)


__all__ = [
    "DoubleDQN",
    "RegularDQN",
    "DuelingDQN",
    "RainbowDQN",
    "dqn_factory"
]
