from typing import Any, Dict, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch
from .supervised import Module as Module_
from torch import Tensor
import torchvision


class Module(Module_):
    def __init__(
        self,
        *,
        optimizer: Union[str, Optimizer],
        scheduler: Union[str, LRScheduler] = None,
        triggers: Dict[str, Dict[str, float]] = None,
        optimizer_kwargs: Dict[str, Any] = None,
        scheduler_kwargs: Dict[str, Any] = None,
        gradient_clip_algorithm: str = None,
        gradient_clip_val: float = None
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            triggers=triggers,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
        )
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


__all__ = ["Module"]
