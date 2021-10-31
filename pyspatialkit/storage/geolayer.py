from pathlib import Path
from abc import ABC, abstractmethod

from .. import DEFAULT_CRS


class GeoLayer(ABC):

    def __init__(self, directory_path: str, *args, **kwargs):
        self.directory_path = Path(directory_path)
        if self.directory_path.is_dir():
            self.load()
        else:
            self.initialize(*args, **kwargs)
            self.directory_path.mkdir(parents=True)
            self.persist()

    def load(self, directory_path: Optional[Path] = None):
        if directory_path is None:
            directory_path = self.directory_path
        self.load_data(directory_path)

    def persist(self, directory_path: Optional[Path] = None):
        if directory_path is None:
            directory_path = self.directory_path
        self.persist_data(directory_path)

    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def persist_data(dir_path: Path):
        raise NotImplementedError

    @abstractmethod
    def load_data(dir_path: Path):
        raise NotImplementedError

    @property
    def name(self):
        return self.folder_path.name

    @property
    def crs(self):
        return DEFAULT_CRS