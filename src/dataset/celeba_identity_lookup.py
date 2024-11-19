"""
File: celeba_identity_lookup.py

This file contains the implementation of the CelebAIdentityLookup class,
which is responsible for building and looking up identities in the CelebA
dataset. The CelebAIdentityLookup class extends the DatasetIdentityLookup
class and provides functionality to create an identity lookup table based
on a specified text file containing identity information.

Libraries and Modules:
- ntpath: Provides functions to work with file paths.
- Path (from pathlib): Represents and manipulates filesystem paths.
- tqdm: Library for displaying progress bars during iteration.
- DatasetIdentityLookup (from stc.dataset.dataset_identity_lookup): Custom ABC module for dataset identity lookup

Usage:
- Create an instance of the CelebAIdentityLookup class by providing the path to a text file containing identity information (`identity_file_path`) and an optional flag to process the test set only (`test_set_only`).
- The CelebAIdentityLookup initializes by building an identity lookup table using the provided text file. It reads the file, extracts file names and identity labels, and populates a dictionary (`identity_dict`) with this information.
- The `lookup` method allows users to look up the identity associated with a given file path or embedding key. If the file path contains a specific string ("___"), it interprets it as an embedding key and extracts the identity label. Otherwise, it uses the base name of the file path to retrieve the identity label from the dictionary.

Attributes:
- None

Note:
- The CelebA dataset is used, and the structure of the text file containing information is assumed to have each line in the format "<file_name> <identity_label>".
- The `test_set_only` flag is used to process the test set exclusively.
- The identity information is stored in the `identity_dict` dictionary for efficient lookups.
"""

import ntpath
from pathlib import Path

from tqdm import tqdm

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup


class CelebAIdentityLookup(DatasetIdentityLookup):
    """
    Class for looking up identity labels in the CelebA dataset

    Attributes:
    - identity_dict (dict): A dictionary to store identity information.
    """

    def __init__(self, identity_file_path: str, test_set_only=False):
        """
        Initialize the CelebAIdentityLookup object.

        Parameters:
        - identity_file_path (str): The file path containing identity information.
        - test_set_only (bool): If True, process the test set only.
        """
        # Initializing an empty dictionary to store identity information
        self.identity_dict = dict()

        # Printing a message indicating the start of building the identity lookup table
        print("Building an identity lookup table for CelebA.")

        # Opening the specified text file containing identity information
        with open(identity_file_path) as txt_file:
            # Reading all lines from the file and creating a list containing each line
            lines = txt_file.readlines()

            # Iterating through each line in the file
            for line in tqdm(lines):
                # Splitting the line into a list of contents using whitespace
                contents = line.split()

                # Extracting the file name and identity label from the contents
                file_name = contents[0]
                id_label = contents[1]

                # Checking if the user wants to process the test set only
                if test_set_only:
                    # Extracting the image index from the file name
                    img_index = int(Path(file_name).stem)

                    # Skipping images that are part of the training set
                    if img_index < 182638:  # The start of the test split
                        continue

                # Adding the file name and identity label to the dictionary
                self.identity_dict[file_name] = id_label

    def lookup(self, file_path: str):
        """
        Look up the identity associated with the file path.

        Parameters:
        - file_path (str): The file path or embedding key.

        Returns:
        - str: The identity label associated with the file.
        """
        # Checking if the file path contains a specific string ("___")
        if "___" in file_path:
            # in case we are passed an embedding key rather than file path
            # Splitting the file path using the specific string
            contents = file_path.split("___")

            # Creating an id_key by appending ".jpg" to the second part of the split
            id_key = f"{contents[1]}.jpg"
        else:
            # If the file path doesn't contain the specific string, extracting the base name
            # else, this is a path to the image (file names are unique in CelebA)
            id_key = ntpath.basename(file_path)

        # Returning the identity label associated with the id_key from the dictionary
        return self.identity_dict[id_key]
