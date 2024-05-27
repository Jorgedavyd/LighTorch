from lightning.pytorch import LightningDataModule
import torch
from lightning.pytorch.cli import LightningCLI


def trainer(
    datamodule: LightningDataModule,
    matmul_precision: str = "high",
    deterministic: bool = True,
    seed: bool | int = 123,
):
    torch.set_float32_matmul_precision(matmul_precision)

    LightningCLI(
        datamodule_class=datamodule,
        seed_everything_default=seed,
        trainer_defaults={
            "deterministic": deterministic,
        },
    )


__all__ = ["trainer"]
