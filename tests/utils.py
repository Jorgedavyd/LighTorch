from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch import Tensor, nn
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import random


def create_inputs(*size) -> Tensor:
    return torch.randn(*size)


class Model(nn.Sequential):
    def __init__(self, in_channels) -> None:
        super().__init__(nn.Linear(in_channels, 10), nn.Linear(10, 1))

    def forward(self, input):
        return super().forward(input)


# Add feature extractor
class Data(Dataset):
    def __init__(
        self,
    ) -> None:
        pass

    def __len__(self) -> int:
        return 100

    def __getitem__(self, index) -> Tensor:
        return torch.randn(10)


class DataModule(LightningDataModule):
    def __init__(self, batch_size: int, pin_memory: bool = False, num_workers: int = 1):
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Data(),
            self.batch_size,
            False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Data(),
            self.batch_size * 2,
            False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def create_mask():
    # Create a rectangle
    n = random.randint(1, 31)  # Random integer for n
    m = random.randint(1, 31)  # Random integer for m
    left = random.randint(0, 32 - n) * 32  # Ensure multiple of 32
    top = random.randint(0, 32 - m) * 32  # Ensure multiple of 32
    width = n * 32
    height = m * 32

    mask = torch.ones(256, 256)
    mask[top : top + height, left : left + width] = 0

    return mask
