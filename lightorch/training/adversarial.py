from typing import Union, Sequence, Any, Tuple, Dict
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from collections import defaultdict
import torch
from torch.optim import Adam, Adadelta, Adamax, AdamW, SGD, LBFGS, RMSprop
from .supervised import Module as Module_
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
    ExponentialLR,
    LinearLR,
)
import torchvision

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


class Module(Module_):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def validation_step(self) -> None:
        grid = torchvision.utils.make_grid(self.sample_imgs[:6])
        self.logger.experiment.add_image("Generator output", grid, 0)
        return super().on_train_epoch_end()

    def training_step(self, batch: Tensor, idx: int) -> Tensor:
        imgs = batch
        # Getting the optimizers
        opt_d, opt_g = self.optimizers()
        # To the latent space
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        self.toggle_optimizer(opt_g)
        # Targets for discriminator
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        # Making the step that minimizes the amount of fake predictions
        g_loss = self.criterion(self.discriminator(self(z)), valid)
        self.log("Generator Loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.toggle_optimizer(opt_d)

        # Real samples
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # Fake samples
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # Mean of both
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

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
                )
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
