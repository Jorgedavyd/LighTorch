from lightning.pytorch import Trainer, LightningModule
from typing import Union, Iterable, Sequence, Any, Tuple, Dict
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from collections import defaultdict

from torch.optim import (
    Adam,
    Adadelta,
    Adamax,
    AdamW,
    SGD,
    LBFGS,
    RMSprop
)

from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
    ExponentialLR,
    LinearLR
)

VALID_OPTIMIZERS = {
    'adam': Adam,
    'adadelta': Adadelta,
    'adamax': Adamax,
    'adamw': AdamW,
    'sgd': SGD,
    'lbfgs': LBFGS,
    'rms': RMSprop
}

VALID_SCHEDULERS = {
    'onecycle': OneCycleLR,
    'plateau': ReduceLROnPlateau,
    'exponential': ExponentialLR,
    'linear': LinearLR
}   

def interval(algo: LRScheduler) -> str:
    if isinstance(algo, OneCycleLR):
        return 'step'
    else:
        return 'epoch'
    
    
class Module(LightningModule):
    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def _default_training_step(self, batch: Tensor, idx: int) -> Tensor:
        x, y = batch
        out_args: Tuple = self(*x)
        return self.compute_loss(*out_args, y)
    
    def training_step(self, batch, idx):
        optimizer, scheduler = self.optimizers()
        ## Computing the loss
        loss = self._default_training_step(batch, idx)
        # Clearing the optimizer
        optimizer.zero_grad()
        # Computing the gradients
        self.manual_backward(loss)
        # Making the grad based step
        optimizer.step()
        # Triggering the scheduler
        scheduler.step()

        if scheduler is not None:
            

    
    def compute_loss(self, *args) -> Tensor | Sequence[Tensor]:
        args = self.criterion(*args)
        self.log_dict({k:v for k, v in zip(self.criterion.labels, args)})
        self.log('hp_metric', Tensor(args[:-1]).sum().item())
        return args[-1]

    def get_param_groups(self, *triggers) -> Tuple:
        """
        Given a list of "triggers", the param groups are defined.
        """

        param_groups: Sequence[Dict[str, Sequence[nn.Module]]] = [defaultdict(list)*len(triggers)]

        for param_group, trigger in zip(param_groups, triggers):
            for name, param in self.named_modules():
                if name.startswith(trigger):
                    param_group['params'].append(param)
            
        return param_groups

    def get_hparams(self) -> Sequence[Dict[str, float]]:
        return [{
            'lr': lr,
            'weight_decay': wd,
            'momentum': mom
        } for lr, wd, mom in zip(self.learning_rate, self.weight_decay, self.momentum)] if getattr(self, 'momentum', False) else [{
            'lr': lr,
            'weight_decay': mom
        } for lr, mom in zip(self.learning_rate, self.weight_decay)]

    def _configure_optimizer(self) -> Optimizer:
        optimizer_args: Dict[str, Union[float, nn.Module]] = []
        for hparam, param_group in zip(self.get_hparams(), self.get_param_groups(*self.triggers)):
            optimizer_args.append(param_group.update(hparam))
        optimizer = VALID_OPTIMIZERS[self.optimizer](optimizer_args)
        return optimizer
    
    def _configure_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        if self.scheduler == 'onecycle':
            return VALID_SCHEDULERS[self.scheduler](optimizer, **self.scheduler_kwargs.update({'total_steps': self.trainer.estimated_stepping_batches}))
        else:
            return VALID_SCHEDULERS[self.scheduler](optimizer, **self.scheduler_kwargs)

    def configure_optimizers(self) -> Optimizer | Sequence[Optimizer]:
        optimizer = self._configure_optimizer()
        scheduler = self._configure_scheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': interval(optimizer),
                'frequency': 1
            }
        }
        
    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
    


__all__ = []