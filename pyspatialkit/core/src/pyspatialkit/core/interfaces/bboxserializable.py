from abc import ABC, abstractmethod, abstractclassmethod
from typing import Optional

import numpy as np

class BBoxSerializable(ABC):

    @abstractmethod
    def to_bytes(self) -> bytes:
        raise NotImplementedError()

    @abstractclassmethod
    def from_bytes(cls, bytes) -> Optional['BBoxSerializable']:
        raise NotImplementedError()

    @abstractmethod
    def is_intersecting_bbox(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_bounds(self)-> np.ndarray:
        raise NotImplementedError()