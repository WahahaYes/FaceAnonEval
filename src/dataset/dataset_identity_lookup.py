from abc import ABC, abstractmethod


class DatasetIdentityLookup(ABC):
    def __init__(self):
        pass

    # looks up the identity corresponding to an image file based on that dataset's specs
    @abstractmethod
    def lookup(self, file_path: str):
        pass
