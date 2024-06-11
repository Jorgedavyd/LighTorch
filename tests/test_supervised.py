# # Future imp
# from lightorch.training.supervised import Module
# from lightorch.nn.criterions import MSELoss
# from lightorch.hparams import htuning
# from .utils import create_inputs, DataModule
# from torch import nn
# import optuna
# import torch
# import random
# from torch import Tensor

# in_size: int = 32
# input: Tensor = create_inputs(1, in_size)
# randint: int = random.randint(-100, 100)
# # Integrated test


# class SupModel(Module):
#     def __init__(self, **hparams) -> None:
#         super().__init__(**hparams)
#         # Criterion
#         self.criterion = MSELoss()

#         self.model = nn.Sequential(
#             nn.Linear(10, 5),
#             nn.ReLU(),
#             nn.Linear(5, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, input: Tensor) -> Tensor:
#         return self.model(input)


# def objective1(trial: optuna.trial.Trial):
#     return dict(
#         triggers = {'model': dict(
#             lr = trial.suggest_float('lr', 1e-4, 1e-1),
#             weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1),
#             momentum = trial.suggest_float('momentum', 0.1, 0.7)
#             )},
#         optimizer = 'sgd',
#         scheduler = 'onecycle',
#         scheduler_kwargs = dict(
#             max_lr = trial.suggest_float('max_lr', 1e-2, 1e-1)
#             ),
#     )

# def objective2(trial: optuna.trial.Trial):
#     return dict(
#         optimizer = torch.optim.Adam,
#         optimizer_kwargs = dict(
#             lr = trial.suggest_float('lr', 1e-4, 1e-1),
#             weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1)
#         ),
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
#         scheduler_kwargs = dict(
#             max_lr = trial.suggest_float('max_lr', 1e-2, 1e-1)
#             )
#     )


# def test_supervised() -> None:
#     htuning(
#         model_class=SupModel,
#         hparam_objective=objective1,
#         datamodule=DataModule,
#         valid_metrics="MSE",
#         datamodule_kwargs=dict(pin_memory=False, num_workers=1, batch_size=1),
#         directions="minimize",
#         precision="high",
#         n_trials=10,
#         trianer_kwargs=dict(fast_dev_run=True, accelerator = 'cpu'),
#     )

#     htuning(
#         model_class=SupModel,
#         hparam_objective=objective2,
#         datamodule=DataModule,
#         valid_metrics="MSE",
#         datamodule_kwargs=dict(pin_memory=False, num_workers=1, batch_size=1),
#         directions="minimize",
#         precision="medium",
#         n_trials=10,
#         trianer_kwargs=dict(fast_dev_run=True, accelerator = 'cpu'),
#     )
