from abc import ABC, abstractmethod

class DataObjectInterface:

    @abstractmethod
    def plot(self, *args, **kwargs):
        raise NotImplementedError()
