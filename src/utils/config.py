from dataclasses import dataclass
from enum import Enum

import yaml
from pydantic import BaseModel


class Networks(str, Enum):
    mlp = "mlp"
    mlp_noisy = "mlp_noisy"
    cnn = "cnn"


class ReplayBufferTypes(str, Enum):
    vanilla = "vanilla"
    prioritized = "prioritized"


class DQNType(str, Enum):
    RegularDQN = "RegularDQN"
    DuelingDQN = "DuelingDQN"
    DoubleDQN = "DoubleDQN"
    RainbowDQN = "RainbowDQN"


class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"
    mps = "mps"


@dataclass
class DefaultValues:
    dqn_type: DQNType = DQNType.RegularDQN
    max_epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    network_type: Networks = Networks.mlp
    replay_buffer_type: ReplayBufferTypes = ReplayBufferTypes.vanilla
    hidden_size: int = 128
    num_hidden_layers: int = 2
    env: str = "MinAtar/Breakout-v1"
    gamma: float = 0.99
    sync_rate: int = 10
    replay_size: int = 1000
    warm_start_steps: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 1000
    episode_length: int = 200
    max_episode_reward: int = 200
    validate_every_n_epochs: int = 100
    rollouts_per_validation: int = 20
    device: str = Device.cpu


class DQNConfig(BaseModel):
    dqn_type: DQNType = DefaultValues.dqn_type
    max_epochs: int = DefaultValues.max_epochs
    batch_size: int = DefaultValues.batch_size
    lr: float = DefaultValues.lr
    network_type: Networks = DefaultValues.network_type
    replay_buffer_type: ReplayBufferTypes = DefaultValues.replay_buffer_type
    hidden_size: int = DefaultValues.hidden_size
    num_hidden_layers: int = DefaultValues.num_hidden_layers
    env: str = DefaultValues.env
    gamma: float = DefaultValues.gamma
    sync_rate: int = DefaultValues.sync_rate
    replay_size: int = DefaultValues.replay_size
    warm_start_steps: int = DefaultValues.warm_start_steps
    eps_start: float = DefaultValues.eps_start
    eps_end: float = DefaultValues.eps_end
    eps_decay: int = DefaultValues.eps_decay
    episode_length: int = DefaultValues.episode_length
    max_episode_reward: int = DefaultValues.max_episode_reward
    validate_every_n_epochs: int = DefaultValues.validate_every_n_epochs
    rollouts_per_validation: int = DefaultValues.rollouts_per_validation
    device: str = DefaultValues.device


def parse_yaml_from_file(file_path: str) -> DQNConfig:
    with open(file_path) as f:
        config = yaml.safe_load(f)

    return DQNConfig(**config)
