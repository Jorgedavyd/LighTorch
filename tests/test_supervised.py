from lightorch.training.supervised import Module
from lightorch.nn.criterions import PeakSignalNoiseRatio
from lightorch.htuning.optuna import htuning
from .utils import Model, create_inputs, DataModule

import random
from torch import Tensor, nn

in_size: int = 32
input: Tensor = create_inputs(1, in_size)
randint: int = random.randint(-100, 100)
# Integrated test


class SupModel(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.criterion = PeakSignalNoiseRatio(1, randint)
        self.model = Model(in_size)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


def objective():
    pass


def test_supervised() -> None:
    htuning(
        model_class=SupModel,
        hparam_objective=objective,
        datamodule=DataModule,
        valid_metrics="MSE",
        datamodule_kwargs=dict(pin_memory=False, num_workers=1, batch_size=1),
        directions="minimize",
        precision="high",
        n_trials=10,
        trianer_kwargs=dict(fast_dev_run=True),
    )
