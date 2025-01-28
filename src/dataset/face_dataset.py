"""
File: face_dataset.py

This file defines a custom PyTorch dataset class, FaceDataset, for
loading face images. It also includes utility functions to create
PyTorch DataLoader, iterator, and enumerator for the dataset.

Libraries and Modules:
- glob: Used for file path pattern matching
- pathlib.Path: Represents and manipulates filesystem path.
- typing.Iterator: Type hint for specifying the type of an iterator.
- cv2: OpenCV library for image processing. 
- torch.utils.data: PyTorch data loading utilities.
- torchvision.transforms: Transformations for image processing.

Usage:
- Use the FaceDataset class to create a custom PyTroch dataset for loading face images from a specified directory.
- The FaceDataset class provides an iterable dataset with optional data transformations.
- Utilize the utility functions create_dataloader, dataset_iterator, and dataset_enumerator to create DataLoader, iterator, and enumerator instances for the dataset.

Attributes:
- None

Note:
- The FaceDataset class is designed to be used as a custom dataset in PyTorch for face image loading tasks.
- It allows customization ofthe file type, data transformations, and inclusion criteria for the CelebA test set.
- The utility functions facilitate the creation of DataLoader, iterator, and enumerator instances for the dataset.
"""

import glob
from pathlib import Path  # Represents and manipulates filesystem paths
from typing import Iterator  # Type hint for specifying the type of an iterator

import cv2  # OpenCV library for image processing
from torch.utils.data import DataLoader, Dataset  # PyTorch data loading utilities
from torchvision import transforms  # Transformations for image processing


class FaceDataset(Dataset):
    """
    Custom PyTorch dataset class for loading face images.

    Attributes:
    - dir (str): The directory containg the face images.
    - filetype (str): The file type of face images (default is ".png")
    - transform (callable): Optional data transformation to be applied on the images.
    - celeba_test_set_only (bool): If True, include only images from the CelebA test set.
    """

    def __init__(
        self,
        dir: str,
        filetype: str = ".png",
        transform=None,
        celeba_test_set_only=False,
    ):
        """
        Initialize the FaceDataset object.

        Parameters:
        - dir (str): The directory containg the face images.
        - filetype (str): The file type of face images (default is "png").
        - transform (callable): Optional data transformations to be applied on the images.
        - celeba_test_set_only (bool): If True, include only images from the CelebA test set.
        """
        # We load the paths to all images at initialization
        self.dir = dir
        self.filetype = filetype
        self.img_paths = []
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

        # This will search any file hierarchy and return all paths ending in <filetype>
        for file in glob.glob(f"{self.dir}//**//*{filetype}", recursive=True):
            if celeba_test_set_only:
                img_index = int(Path(file).stem)
                if img_index < 182638:  # The start of the test split
                    continue
            self.img_paths.append(file)

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
        - int: The number of images in the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Get the images and its path at the specified index.

        Parameters:
        - idx (int): The index of the image in the dataset.

        Returns:
        - tuple: A tuple containg the image and its path
        """
        img_path = self.img_paths[idx]

        # Read the image (note that OpenCV returns a uint8 image [0-255] in BGR channels)
        img = cv2.imread(img_path)
        # Our transform will convert to CxWxH tensor in [0, 1] range
        img = self.transform(img)

        return img, img_path


def create_dataloader(
    dataset: Dataset, batch_size: int | None = 1, shuffle: bool | None = None
):
    """
    Create a PyTorch DataLoader for the given dataset.

    Parameters:
    - dataset (Dataset): The dataset to create a DataLoader for.
    - batch_size (int): The batch size of the DataLoader (default is 1).
    - shuffle (bool): Whether to shuffle the data in the DataLoader (default is None).

    Returns:
    - DataLoader: The created DataLoader.
    """
    return DataLoader(dataset, batch_size, shuffle)


def dataset_iterator(
    dataset: Dataset, batch_size: int | None = 1, shuffle: bool | None = None
) -> Iterator:
    """
    Create an iterator for the given dataset.

    Parameters:
    - dataset (Dataset): The dataset to create an iterator for.
    - batch_size (int): The batch size for the iterator (default is 1).
    - shuffle (bool): Whether to shuffle the data in the iterator (default is None).

    Returns:
    - Iterator: The created iterator.
    """
    return iter(create_dataloader(dataset, batch_size, shuffle))


def dataset_enumerator(
    dataset: Dataset, batch_size: int | None = 1, shuffle: bool | None = None
) -> Iterator:
    """
    Create an enumerator for the given dataset.

    Parameters:
    - dataset (Dataset): The dataset to create an enumerator for.
    - batch_size (int): The batch size for the enumerator (default is 1).
    - shuffle (bool): Whether to shuffle the data in the enumerator (default is None).

    Returns:
    - Iterator: The created enumerator
    """
    return enumerate(create_dataloader(dataset, batch_size, shuffle))
