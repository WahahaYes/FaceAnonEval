from pathlib import Path

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup


class LFWIdentityLookup(DatasetIdentityLookup):
    def __init__(self):
        pass

    def lookup(self, file_path: str):
        # LFW is simple, the identities are stored in a labeled folder
        return Path(file_path).parent.stem
