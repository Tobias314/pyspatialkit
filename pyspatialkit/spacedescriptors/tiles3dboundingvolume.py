from typing import Dict, List
from abc import ABC, abstractmethod


class Tiles3dBoundingVolume(ABC):
    
    @abstractmethod
    def to_tiles3d_dict(self) -> Dict[str: List[float]]:
        raise NotImplementedError