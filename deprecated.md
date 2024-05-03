# Deprecated module
This module is deprecated because of the object quality, still useful.

 ```python 
from deprecated import TrainingPhase
from typing import Union, Tuple
from torch import nn, Tensor
from utils import get_default_device

class Model(TrainingPhase):
    def __init__(self, name_run: str, criterion, *args, **kwargs) -> None:
        super().__init__(name_run, criterion)
    def forward(self, ...) -> Union[Tensor, Tuple[Tensor, ...]]:
        ...
    def compute_loss_train(self, batch: Tensor) -> Tuple[Dict[str, float], Tensor]:
        ### Compute the metrics that will be logged, 
        ### also return the overall loss that will 
        ### be backpropagated. This is for personalized
        ### training step

    def compute_val_metrics(self, batch: Tensor) -> Dict[str, float]:
        ### Compute the validation metrics
        ### and return them in dictionary

# Hyperparameters
device = get_default_device()
epochs: int = 10
lr: Union[List[float], float] = 1e-2
weight_decay: Union[List[float], float] = 1e-4
batch_size: int = 3
grad_clip: Union[List[float], float] = 3

saving_div = 5 # Save checkpoint every 5 steps

train_loader = ...
val_loader = ...
"""
modules: List[str] = [['conv1', 'conv2'], ['fc']] #example (for different param groups or even optimizers)
"""
optimizer: Union[List[any], any] = torch.optim.Adam # We can use different optimizers for certain parts of the model
scheduler: Union[List[any], any] = torch.optim.lr_scheduler.OneCycleLR

graph: bool = True
sample_tensor = torch.randn(1, 1024,1024).to(device)

self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        batch_size: int,
        lr: Union[float, Iterable[float]],
        weight_decay: Union[float, Iterable[float]] = 0.0,
        grad_clip: bool = False,
        opt_func: Any = torch.optim.Adam,
        lr_sched: Any = None,
        saving_div: int = 5,
        graph: bool = False,
        sample_input: Tensor = None,
        modules: Iterable = None,

if __name__ == '__main__':
    model = Model(...)
    model.fit(
        train_loader,
        val_loader,
        epochs,
        batch_size,
        lr,
        weight_decay,
        grad_clip,
        opt_func,
        lr_sched,
        saving_div,
        graph,
        sample_input,
        #modules
    )
 ```