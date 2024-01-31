import glob
from pathlib import Path
from typing import Iterator

import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(
        self,
        dir: str,
        filetype: str = ".png",
        transform=None,
        celeba_test_set_only=False,
    ):
        # We load the paths to all images at initialization
        self.dir = dir
        self.img_paths = []
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

        # This will search any file hierarchy and return all paths ending in <filetype>
        for file in glob.glob(f"{self.dir}//**//*{filetype}", recursive=True):
            if celeba_test_set_only:
                img_index = int(Path(file).stem)
                if img_index < 182638:  # this is the start of the test split
                    continue
            self.img_paths.append(file)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # read the image (note that opencv returns a uint8 image [0-255] in BGR channels)
        img = cv2.imread(img_path)
        # our transform will convert to CxWxH tensor in [0, 1] range
        img = self.transform(img)

        return img, img_path


def create_dataloader(
    dataset: Dataset, batch_size: int | None = 1, shuffle: bool | None = None
):
    return DataLoader(dataset, batch_size, shuffle)


def dataset_iterator(
    dataset: Dataset, batch_size: int | None = 1, shuffle: bool | None = None
) -> Iterator:
    return iter(create_dataloader(dataset, batch_size, shuffle))


def dataset_enumerator(
    dataset: Dataset, batch_size: int | None = 1, shuffle: bool | None = None
) -> Iterator:
    return enumerate(create_dataloader(dataset, batch_size, shuffle))
