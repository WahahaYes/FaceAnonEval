from pathlib import Path  # Represents and manipulates filesystem paths

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup


class CodecIdentityLookup(DatasetIdentityLookup):
    """
    Class for looking up identities in the codec dataset.

    Attributes:
    - None
    """

    def __init__(self):
        """
        Initialize.

        Parameters:
        - None
        """
        pass

    def lookup(self, file_path: str):
        """
        Look up the identitiy associated with the file path.

        Parameters:
        - file_path (str): The file path or embedding key.

        Returns:
        - str: The identity label assocated with the file
        """
        if "___" in file_path:
            # In case we are passed an embedding key rather than file path
            contents = file_path.split("___")
            return contents[1]
        else:
            return Path(file_path).stem
