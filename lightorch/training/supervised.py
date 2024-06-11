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
    """
    init:
        triggers: Dict[str, Dict[str, float]] -> This is an
        interpretative implementation for grouped optimization
        where the parameters are stored in groups given a "trigger",
        namely, as trigger parameters you can put a string describing
        the beginning of the parameters to optimize in a group.
        optimizer: str | Optimizer -> Name of the optimizer or an Optimizer instance.
        scheduler: str | LRScheduler -> Name of the scheduler or a Scheduler instance.
        scheduler_kwargs: Dict[str, Any] -> Arguments of the scheduler.
        gradient_clip_algorithm: str -> Gradient clip algorithm [value, norm].
        gradient_clip_val: float -> Clipping value.
    """

    def __init__(
        self,
        *,
        optimizer: Union[str, Optimizer],
        scheduler: Union[str, LRScheduler] = None,
        triggers: Dict[str, Dict[str, float]] = None,
        optimizer_kwargs: Dict[str, Any] = None,
        scheduler_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        for att in kwargs:
            setattr(self, att, kwargs[att])

        # Initializing the optimizer and the triggers
        self.triggers = triggers
        if triggers is not None:
            assert (
                optimizer_kwargs is None
            ), "Not valid optimizer_kwargs parameter for trigger-based setting, include all optimizer parameters in the dictionary with their respective name."
            self.triggers = triggers
        else:
            if not isinstance(optimizer, Optimizer):
                assert (
                    optimizer_kwargs is not None
                ), "Must specify optimizer_kwargs parameter for non-trigger-based setting."
                self.optimizer_kwargs = optimizer_kwargs
            else:
                assert (
                    optimizer_kwargs is None
                ), "Not valid optimizer_kwargs parameter for initialized optimizer."
                self.optimizer = optimizer

        if isinstance(optimizer, str) or issubclass(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            if not getattr(self, "optimizer", False):
                raise ValueError(
                    f"Not valid optimizer parameter, expecting str | Optimizer got {type(optimizer)}"
                )

        # Initializing the scheduler
        if scheduler is not None:
            if isinstance(scheduler, str):
                self.scheduler = scheduler
                self.scheduler_kwargs = scheduler_kwargs
            elif isinstance(scheduler, LRScheduler):
                self.scheduler = lambda optimizer: scheduler(
                    optimizer=optimizer, **scheduler_kwargs
                )
            else:
                raise ValueError("Not valid scheduler parameter")
        else:
            assert (
                scheduler_kwargs is None
            ), "Not valid scheduler_kwargs parameter for NoneType scheduler"
            self.scheduler = None

    def loss_forward(self, batch: Tensor, idx: int) -> Dict[str, Union[Tensor, float]]:
        raise NotImplementedError("Should have defined loss_forward method.")

    def training_step(self, batch: Tensor, idx: int) -> Tensor:
        kwargs = self.loss_forward(batch, idx)
        return self._compute_training_loss(**kwargs)

    @torch.no_grad()
    def validation_step(self, batch: Tensor, idx: int) -> None:
        kwargs = self.loss_forward(batch, idx)
        return self._compute_valid_metrics(**kwargs)

    def _compute_training_loss(self, **kwargs) -> Union[Tensor, Sequence[Tensor]]:
        args = self.criterion(**kwargs)
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
    def _compute_valid_metrics(self, **kwargs) -> None:
        args = self.criterion(**kwargs)
        self.log_dict(
            {f"Validation/{k}": v for k, v in zip(self.criterion.labels, args)},
            True,
            True,
            True,
            True,
        )

    def get_param_groups(self) -> Tuple:
        """
        Given a list of "triggers", the param groups are defined.
        """
        if self.triggers is not None:
            param_groups: Sequence[Dict[str, Sequence[nn.Module]]] = [
                defaultdict(list) for _ in range(len(self.triggers))
            ]
            # Update the model parameters per group and finally add the
            # hyperparameters
            for param_group, trigger in zip(param_groups, self.triggers):
                for name, param in self.named_parameters():
                    if name.startswith(trigger):
                        param_group["params"].append(param)

                param_group.update(self.triggers[trigger])

            return param_groups
        return None

    def _configure_optimizer(self) -> Optimizer:
        if params := self.get_param_groups() is not None:
            if isinstance(self.optimizer, str):
                return VALID_OPTIMIZERS[self.optimizer](params)
            elif isinstance(self.optimizer, torch.optim.Optimizer):
                return self.optimizer
            elif issubclass(self.optimizer, torch.optim.Optimizer):
                return self.optimizer(params)
        else:

            if isinstance(self.optimizer, str):
                self.optimizer = VALID_OPTIMIZERS[self.optimizer]
            elif isinstance(self.optimizer, Optimizer):
                return self.optimizer
            elif issubclass(self.optimizer, Optimizer):
                pass

            return self.optimizer(self.parameters(), **self.optimizer_kwargs)

    def _configure_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        if isinstance(self.scheduler, str):
            if self.scheduler == "onecycle":
                self.scheduler_kwargs["total_steps"] = (
                    self.trainer.estimated_stepping_batches
                )
            return VALID_SCHEDULERS[self.scheduler](optimizer, **self.scheduler_kwargs)
        else:
            return self.scheduler(optimizer)

    def configure_optimizers(self) -> Union[Optimizer, Sequence[Optimizer]]:
        optimizer = self._configure_optimizer()
        if self.scheduler is not None:
            scheduler = self._configure_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": interval(scheduler),
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


__all__ = ["Module"]
