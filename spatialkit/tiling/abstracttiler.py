from abc import ABC, abstractmethod
from typing import Iterable, List

class AbstractTiler(ABC):

    @abstractmethod
    def __iter__(self) -> Iterable:
        raise NotImplementedError()

    @abstractmethod
    def partition(self, num_partitions: int) -> List['AbstractTiler']:
        raise NotImplementedError()