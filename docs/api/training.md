# Supervised

```python 
from lightorch.training.supervised import Module


class Model(Module):
    def __init__(*model_args, **hparams) -> None:
        super().__init__(**hparams)
        ## Define model backbone with *model_args

    def forward(self, input: Tensor) -> Tensor:
        ...
    
    def loss_forward(self, batch: Tensor, idx: int) -> Tensor:
        input, target = batch #example
        # Compute the arguments of self.criterion
        # Example 1:
        # criterion: Callable[[Tensor, Tensor], Tensor] = nn.MSELoss() # arguments: input, target
        # Example 2:
        # elbo_loss: nn.Module = ELBO(*args) # arguments: input, target, mu, logvar 
        out = self(...)
        return out, target
```
# Adversarial

```python 
from lightorch.training.adversarial import Module


class Model(Module):
    def __init__(*model_args, **hparams) -> None:
        super().__init__(**hparams)
        ## Define model backbone with *model_args

    def forward(self, input: Tensor) -> Tensor:
        ...
    
    def loss_forward(self, batch: Tensor, idx: int) -> Tensor:
        input, target = batch #example
        # Compute the arguments of self.criterion
        # Example 1:
        # criterion: Callable[[Tensor, Tensor], Tensor] = nn.MSELoss() # arguments: input, target
        # Example 2:
        # elbo_loss: nn.Module = ELBO(*args) # arguments: input, target, mu, logvar 
        out = self(...)
        return out, target
```

# CLI

## `training.py` file:

```python 
from lightorch.training.cli import trainer
from ... import DataModule
from ... import Model

if __name__ == '__main__':
    trainer(
        matmul_precision = 'medium', # default
        deterministic = True, # default
        seed = 123, # default
    )

```

## `config.yaml` file:

```yaml 
trainer: # trainer arguments
  logger: true 
  enable_checkpointing: true
  max_epochs: 250
  accelerator: cuda
  devices:  1
  precision: 32
  
model:
  class_path: utils.FourierVAE #model relative path
  dict_kwargs: #**hparams
    encoder_lr: 2e-2
    encoder_wd: 0
    decoder_lr: 1e-2
    decoder_wd: 0
    alpha:
      - 0.02
      - 0.003
      - 0.003
      - 0.01
    beta: 0.00001
    optimizer: adam

data: # Dataset arguments
  class_path: data.DataModule
  init_args:
    type_dataset: mnist 
    batch_size: 12
    pin_memory: true
    num_workers: 8
    

```
