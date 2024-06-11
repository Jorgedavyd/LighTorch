from lightning.pytorch.cli import LightningCLI
from typing import Union
import torch


def trainer(
    matmul_precision: str = "high",
    deterministic: bool = True,
    seed: Union[bool, int] = 123,
):
    torch.set_float32_matmul_precision(matmul_precision)

    LightningCLI(
        seed_everything_default=seed,
        trainer_defaults={
            "deterministic": deterministic,
        },
    )


__all__ = ["trainer"]
