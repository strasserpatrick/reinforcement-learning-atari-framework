from typing import Iterator, Tuple

from torch.utils.data import IterableDataset

from core.replay_memory import AbstractReplayMemory


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: AbstractReplayMemory, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        batch = self.buffer.sample(
            self.sample_size
        )
        batch_size = len(batch[0])
        feature_size = len(batch)

        for i in range(batch_size):
            yield tuple([batch[j][i] for j in range(feature_size)])
