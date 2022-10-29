from abc import ABC, abstractmethod
from typing import Optional

from ..crs.geocrs import GeoCrs
from ..crs.geocrstransformer import GeoCrsTransformer

class GeoDataObjectInterface:

    @abstractmethod
    def plot(self, *args, **kwargs):
        raise NotImplementedError()
