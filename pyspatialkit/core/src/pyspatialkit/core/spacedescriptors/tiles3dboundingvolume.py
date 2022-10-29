from typing import Dict, List
from abc import ABC, abstractmethod


class Tiles3dBoundingVolume(ABC):
    
    @abstractmethod
    def to_tiles3d_bounding_volume_dict(self) -> Dict[str, List[float]]:
        raise NotImplementedError