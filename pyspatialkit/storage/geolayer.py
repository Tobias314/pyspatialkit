from typing import Optional
from pathlib import Path
from abc import ABC, abstractmethod
import shutil

from ..globals import DEFAULT_CRS

class GeoLayerOwner(ABC):
    @abstractmethod
    def on_child_delete(child: 'GeoLayer'):
        raise NotImplementedError

class GeoLayer(ABC):

    def __init__(self, directory_path: str, *args, **kwargs):
        self.directory_path = Path(directory_path)
        self.owner: Optional[GeoLayerOwner] = None
        if self.directory_path.is_dir():
            self.load()
        else:
            self.directory_path.mkdir(parents=True)
            self.initialize(*args, **kwargs)
            self.persist()

    def load(self, directory_path: Optional[Path] = None):
        if directory_path is None:
            directory_path = self.directory_path
        self.load_data(directory_path)

    def persist(self, directory_path: Optional[Path] = None):
        if directory_path is None:
            directory_path = self.directory_path
        self.persist_data(directory_path)

    def register_owner(self, owner: GeoLayerOwner):
        self.owner = owner

    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def persist_data(self, dir_path: Path):
        raise NotImplementedError

    @abstractmethod
    def load_data(self, ir_path: Path):
        raise NotImplementedError

    def _delete_permanently(self):
        pass

    def delete_permanently(self):
        if self.owner is not None:
            self.owner.on_child_delete(self)
        self._delete_permanently()
        shutil.rmtree(self.directory_path)

    @property
    def name(self):
        return self.directory_path.name

    @property
    def crs(self):
        return DEFAULT_CRS