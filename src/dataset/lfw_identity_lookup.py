from pathlib import Path

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup


class LFWIdentityLookup(DatasetIdentityLookup):
    def __init__(self):
        pass

    def lookup(self, file_path: str):
        if "___" in file_path:
            # in case we are passed an embedding key rather than file path
            contents = file_path.split("___")
            return contents[0]
        else:
            # LFW is simple, the identities are stored in a labeled folder
            return Path(file_path).parent.stem
