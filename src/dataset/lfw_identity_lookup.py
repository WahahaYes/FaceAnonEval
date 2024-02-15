"""
File: lfw_identity_lookup.py

This file defines a class, LFWIdentityLookup, for looking up identities
in the Labeled Faces in the Wild (LFW) dataset. It is designed to be
used as part of a larger face recognition system.

Libraries and Modules:
- pathlib.Path: Represents and manipulates filesystem paths.
- src.dataset.dataset_identity_lookup: Custom module providing DatasetIdentityLookup ABC.

Usage:
- Use the LFWIdentityLookup class to create an identity lookup for the LFW dataset.
- This class provides a method, lookup, to retrieve the identity label asspcated with a given file path or embedding key.

Attributes:
- None

Note:
- The LFWIdentityLookup class is specific to the Labeled Faces in the Wild (LFW) dataset and inherits the general identity lookup functionality from DatasetIdentityLookup.
- The lookup method extracts the identity label from either the file path or an embedding key based on the presence of the "___" string.
- In the case of LFW, identities are assumed to be stored in labeled folders, making it simple to retrieve the identity label.
"""

from pathlib import Path  # Represents and manipulates filesystem paths

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup


class LFWIdentityLookup(DatasetIdentityLookup):
    """
    Class for looking up identities in the Labled Faces in the Wild (LFW) dataset.

    Attributes:
    - None
    """

    def __init__(self):
        """
        Initialize the LFWIdentityLookup object.

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
            return contents[0]
        else:
            # LFW is simple, the identities are stored in a labeled folder
            return Path(file_path).parent.stem
