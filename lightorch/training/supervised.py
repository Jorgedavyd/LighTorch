from lightning.pytorch import LightningModule
from typing import Union, Sequence, Any, Tuple, Dict
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from collections import defaultdict
import torch
from torch.optim import Adam, Adadelta, Adamax, AdamW, SGD, LBFGS, RMSprop

from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
    ExponentialLR,
    LinearLR,
)

VALID_OPTIMIZERS = {
    "adam": Adam,
    "adadelta": Adadelta,
    "adamax": Adamax,
    "adamw": AdamW,
    "sgd": SGD,
    "lbfgs": LBFGS,
    "rms": RMSprop,
}

VALID_SCHEDULERS = {
    "onecycle": OneCycleLR,
    "plateau": ReduceLROnPlateau,
    "exponential": ExponentialLR,
    "linear": LinearLR,
}


def interval(algo: LRScheduler) -> str:
    if isinstance(algo, OneCycleLR):
        return "step"
    else:
        return "epoch"


class Module(LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for att in kwargs:
            setattr(self, att, kwargs[att])
        # Setting up the gradient clipping
        self.trainer.gradient_clip_algorithm = self.gradient_clip_algorithm
        self.trainer.gradient_clip_val = self.gradient_clip_val

    def training_step(self, batch: Tensor, idx: int) -> Tensor:
        args = self.loss_forward(batch, idx)
        return self._compute_training_loss(*args)

    @torch.no_grad()
    def validation_step(self, batch: Tensor, idx: int) -> None:
        args = self.loss_forward(batch, idx)
        return self._compute_valid_metrics(*args)

    def _compute_training_loss(self, *args) -> Tensor | Sequence[Tensor]:
        args = self.criterion(*args)
        self.log_dict(
            {f"Training/{k}": v for k, v in zip(self.criterion.labels, args)},
            True,
            True,
            True,
            True,
        )
        self.log("hp_metric", sum(args[:-1]), True, True, True, True)
        return args[-1]

    @torch.no_grad()
    def _compute_valid_metrics(self, *args) -> None:
        args = self.criterion.val_step(*args)
        self.log_dict(
            {f"Validation/{k}": v for k, v in zip(self.criterion.val_labels, args)},
            True,
            True,
            True,
            True,
        )

    def get_param_groups(self, *triggers) -> Tuple:
        """
        Given a list of "triggers", the param groups are defined.
        """

        param_groups: Sequence[Dict[str, Sequence[nn.Module]]] = [
            defaultdict(list) * len(triggers)
        ]

        for param_group, trigger in zip(param_groups, triggers):
            for name, param in self.named_modules():
                if name.startswith(trigger):
                    param_group["params"].append(param)

        return param_groups

    def get_hparams(self) -> Sequence[Dict[str, float]]:
        return (
            [
                {"lr": lr, "weight_decay": wd, "momentum": mom}
                for lr, wd, mom in zip(
                    self.learning_rate, self.weight_decay, self.momentum
                )
            ]
            if getattr(self, "momentum", False)
            else [
                {"lr": lr, "weight_decay": mom}
                for lr, mom in zip(self.learning_rate, self.weight_decay)
            ]
        )

    def _configure_optimizer(self) -> Optimizer:
        optimizer_args: Dict[str, Union[float, nn.Module]] = []
        for hparam, param_group in zip(
            self.get_hparams(), self.get_param_groups(*self.triggers)
        ):
            optimizer_args.append(param_group.update(hparam))
        optimizer = VALID_OPTIMIZERS[self.optimizer](optimizer_args)
        return optimizer

    def _configure_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        if self.scheduler == "onecycle":
            return VALID_SCHEDULERS[self.scheduler](
                optimizer,
                **self.scheduler_kwargs.update(
                    {"total_steps": self.trainer.estimated_stepping_batches}
                ),
            )
        else:
            return VALID_SCHEDULERS[self.scheduler](optimizer, **self.scheduler_kwargs)

    def configure_optimizers(self) -> Optimizer | Sequence[Optimizer]:
        optimizer = self._configure_optimizer()
        scheduler = self._configure_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval(optimizer),
                "frequency": 1,
            },
        }


__all__ = ["Module"]
