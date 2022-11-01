from abc import ABC, abstractmethod, abstractclassmethod
from typing import Optional

import numpy as np

class BBoxSerializable(ABC):

    @abstractmethod
    def to_bytes(self) -> bytes:
        raise NotImplementedError()

    @abstractclassmethod
    def from_bytes(cls, data: bytes, bbox: Optional[np.ndarray]) -> Optional['BBoxSerializable']:
        raise NotImplementedError()

    @abstractmethod
    def is_intersecting_bbox(self, bbox: np.ndarray) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_bounds(self)-> np.ndarray:
        raise NotImplementedError()