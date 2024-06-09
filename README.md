![status](https://img.shields.io/badge/status-beta-red.svg)
[![pypi](https://img.shields.io/pypi/v/lightorch)](https://pypi.org/project/lightorch)
![CI](https://github.com/Jorgedavyd/LighTorch/actions/workflows/CI.yml/badge.svg)
![CD](https://github.com/Jorgedavyd/LighTorch/actions/workflows/CD.yml/badge.svg)
[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code-style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# LighTorch

<p align="center">
  <img src="https://raw.githubusercontent.com/Jorgedavyd/LighTorch/main/docs/source/logo.png" height = 350 width = 350 />
</p>

A Pytorch and Lightning based framework for research and ml pipeline automation.

## Framework
1. $\text{Hyperparameter space}.$
2. $\text{Genetic algorithms(single-objective/multi-objective)}$
3. $\text{Best hyperparameters in config.yaml}$
4. $\text{Training session}$

### htuning.py
```python
from lightorch.htuning.optuna import htuning
from ... import NormalModule
from ... import FourierVAE

def objective(trial) -> Dict[str, float]:
    ... # define hyperparameters
    return hyperparameters

if __name__ == '__main__':
    htuning(
        model_class = FourierVAE,
        hparam_objective = objective,
        datamodule = NormalModule,
        valid_metrics = [f"Training/{name}" for name in [
            "Pixel",
            "Perceptual",
            "Style",
            "Total variance",
            "KL Divergence"]],
        directions = ['minimize', 'minimize', 'minimize', 'minimize', 'minimize'],
        precision = 'medium',
        n_trials = 150,
    )
```
exec: `python3 -m htuning`

### config.yaml
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

### training.py
```python
from lightorch.training.cli import trainer

if __name__ == '__main__':
    trainer()
```
exec: `python3 -m training -c config.yaml`


## Features
- Built in Module class for:
    - Adversarial training.
    - Supervised, Self-supervised training.
- Multi-Objective and Single-Objective optimization and Hyperparameter tuning with optuna.

## Modules
- Fourier Convolution.
- Fourier Deconvolution.
- Partial Convolution. (Optimized implementation)
- Grouped Query Attention, Multi Query Attention, Multi Head Attention. (Interpretative usage) (with flash-attention option)
- Self Attention, Cross Attention.
- Normalization methods.
- Positional encoding methods.
- Embedding methods.
- Useful criterions.
- Useful utilities.
- Built-in Default Feed Forward Networks.
- Adaptation for $\mathbb{C}$ modules.
- Interpretative Deep Neural Networks.
- Monte Carlo forward methods.

## Contact  

- [Linkedin](https://www.linkedin.com/in/jorge-david-enciso-mart%C3%ADnez-149977265/)
- [GitHub](https://github.com/Jorgedavyd)
- Email: jorged.encyso@gmail.com

## Citation

```
@misc{lightorch,
  author = {Jorge Enciso},
  title = {LighTorch: Automated Deep Learning framework for researchers},
  howpublished = {\url{https://github.com/Jorgedavyd/LighTorch}},
  year = {2024}
}
```
