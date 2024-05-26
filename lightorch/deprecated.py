from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Any, Iterable, Union
from collections import defaultdict
import torch
import os
from torch import Tensor, nn
from copy import deepcopy
from tqdm import tqdm

# REASON: Optimization performance against other libraries (lightning)


def create_config(name_run: str):
    os.makedirs(f"./{name_run}/models", exist_ok=True)
    return {
        "name": name_run,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epoch": 0,
        "global_step_train": 0,
        "global_step_val": 0,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
    }


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class TrainingPhase(nn.Module):
    def __init__(self, name_run: str, criterion) -> None:
        super().__init__()
        self.name_run = name_run
        self.config = create_config(name_run)
        # Tensorboard writer
        self.writer: SummaryWriter = SummaryWriter(f"{name_run}")
        self.name: str = name_run
        self.criterion = criterion
        # History of changes
        self.train_epoch: List[List[int]] = []
        self.val_epoch: List[List[int]] = []
        self.lr_epoch: Dict[str, List[float]] = defaultdict(list)
        self.num_runs: int = 0
        os.makedirs(f"{self.name_run}/images/", exist_ok=True)

    @torch.no_grad()
    def batch_metrics(self, metrics: dict, card: str) -> None:

        if card == "Training" or "Training" in card.split("/"):
            self.train()
            mode = "Training"
        elif card == "Validation" or "Validation" in card.split("/"):
            self.eval()
            mode = "Validation"
        else:
            raise ValueError(
                f"{card} is not a valid card, it should at least have one of these locations: [Validation, Training]"
            )

        self.writer.add_scalars(
            f"{card}/metrics",
            metrics,
            (
                self.config["global_step_train"]
                if mode == "Training"
                else self.config["global_step_val"]
            ),
        )
        self.writer.flush()

        if mode == "Training":
            self.train_epoch.append(list(metrics.values()))
        elif mode == "Validation":
            self.val_epoch.append(list(metrics.values()))

    @torch.no_grad()
    def validation_step(self, batch) -> None:
        metrics = self.compute_val_metrics(batch)
        self.batch_metrics(metrics, "Validation")
        self.config["global_step_val"] += 1

    def training_step(self, batch) -> None:
        # Compute loss and metrics
        metrics, loss = self.compute_train_loss(batch)
        # Write metrics to TensorBoard
        if metrics is not None:
            self.batch_metrics(metrics, "Training")
        # Backpropagation
        loss.backward()
        # Gradient clipping
        if self.grad_clip:
            for grad_clip, params in zip(self.grad_clip, self.params):
                nn.utils.clip_grad_value_(params, grad_clip)
        # Optimizer and scheduler steps
        for optimizer, scheduler in zip(self.optimizer, self.scheduler):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                for param_group in optimizer.param_groups:
                    lr = param_group["lr"]
                    idx = self.optimizer.index(optimizer)
                    self.lr_epoch[f"{optimizer.__class__.__name__}_{idx}"].append(lr)
                    self.writer.add_scalars(
                        "Training",
                        {f"lr_{optimizer.__class__.__name__}_{idx}": lr},
                        self.config["global_step_train"],
                    )
                    scheduler.step()
        # Increment global step
        self.config["global_step_train"] += 1
        self.writer.flush()

    @torch.no_grad()
    def end_of_epoch(self) -> None:
        train_metrics = Tensor(self.train_epoch).mean(dim=-1)
        val_metrics = Tensor(self.val_epoch).mean(dim=-1)
        lr_epoch = {
            key: Tensor(value).mean().item() for key, value in self.lr_epoch.items()
        }

        train_metrics = {
            key: value for key, value in zip(self.criterion.labels, train_metrics)
        }
        val_metrics = {
            key: value for key, value in zip(self.criterion.labels, val_metrics)
        }

        self.writer.add_scalars(
            "Training/Epoch", train_metrics, global_step=self.config["epoch"]
        )
        self.writer.add_scalars(
            "Validation/Epoch", val_metrics, global_step=self.config["epoch"]
        )

        hparams = {"Epochs": self.config["epoch"], "Batch Size": self.batch_size}
        # Criterion factors
        hparams.update(self.criterion.factors)
        # Learning rate
        hparams.update(lr_epoch)
        # Weight decay
        hparams.update(
            {
                f"weight_decay_{optimizer.__class__.__name__}_{idx}": param_group[
                    "weight_decay"
                ]
                for optimizer in self.optimizer
                for idx, param_group in enumerate(optimizer.param_groups)
            }
        )
        # Gradient clipping
        hparams.update(
            {
                f"grad_clip_{optimizer.__class__.__name__}": clip
                for optimizer, clip in zip(self.optimizer, self.grad_clip)
            }
        )
        self.writer.add_hparams(hparams, train_metrics)

        self.writer.flush()

        self.train_epoch = []
        self.val_epoch = []
        self.lr_epoch = defaultdict(list)

    def save_config(self) -> None:
        os.makedirs(f"{self.name}/models", exist_ok=True)
        # Model weights
        self.config["optimizer_state_dicts"] = [
            deepcopy(optimizer.state_dict()) for optimizer in self.optimizer
        ]
        if self.scheduler is not None:
            self.config["scheduler_state_dicts"] = [
                deepcopy(scheduler.state_dict()) for scheduler in self.scheduler
            ]
        self.config["model_state_dict"] = deepcopy(self.state_dict())

        torch.save(
            self.config,
            f'./{self.name}/models/{self.config["name"]}_{self.config["epoch"]}.pt',
        )

    @torch.no_grad()
    def load_config(self, epochs, train_loader) -> None:
        try:
            root_path = f"./{self.name_run}/models"
            files = sorted(os.listdir(root_path))
            if len(files) == 0:
                raise FileNotFoundError
            for idx, file in enumerate(files):
                print(f"{idx+1}. {file}")
            config = int(input("Choose the config: ")) - 1
        except (ValueError, FileNotFoundError):
            return None

        config: Dict[str, Any] = torch.load(os.path.join(root_path, files[config]))
        # Modules
        self.load_state_dict(config["model_state_dict"])
        # Optimizer
        for optimizer, state in zip(self.optimizer, config["optimizer_state_dict"]):
            optimizer.load_state_dict(state)
        # Scheduler
        if self.scheduler is not None and config["scheduler_state_dict"] is not None:
            for scheduler, state in zip(self.scheduler, config["scheduler_state_dict"]):
                state["total_steps"] += epochs * int(len(train_loader))
                scheduler.load_state_dict(state)
        # Other parameters
        self.config["epoch"] = config["epoch"]
        self.config["device"] = config["device"]
        self.config["global_step_train"] = config["global_step_train"]
        self.config["global_step_val"] = config["global_step_val"]

    scheduler = [None]

    def get_module_parameters(
        modules: Iterable[Union[nn.Module, str]],
        hyperparams: Iterable[Dict[str, float]],
    ):
        assert len(modules) == len(hyperparams)
        return [
            {"params": module}.update(hyperparam)
            for module, hyperparam in zip(modules, hyperparams)
        ]

    def fit(
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
    ) -> None:
        self.num_runs += 1
        torch.cuda.empty_cache()
        # Given a list of learning rates
        if isinstance(lr, list):
            assert isinstance(
                weight_decay, (tuple, list, set)
            ), "Must define the regularization term for every param group or optimizer"
            assert len(lr) == len(
                weight_decay
            ), "Should have the same amount of learning rates and regularization parameters"
            if grad_clip:
                assert isinstance(
                    grad_clip, (tuple, list, set)
                ), "You should define the gradient clipping term for every param group or optimizer"
                assert len(grad_clip) == len(
                    lr
                ), "Should have the same amount of learning rates and gradient clipping parameters"
            assert (
                modules is not None
            ), "Should define by param groups with modules argument for different learning rates or optimizers"
            assert len(modules) > 1, "Should have more than one module"
            if isinstance(opt_func, (tuple, list, set)):
                if isinstance(lr_sched, (tuple, list, set)):
                    assert (
                        len(lr_sched) == len(opt_func) == len(lr)
                        or len(lr_sched) == 1
                        or lr_sched is None
                    ), "Learning rate scheduler not correctly defined"
            else:
                assert lr_sched is None or not isinstance(lr_sched, (tuple, list, set))

        # If it's the first run on notebook, we need to define and load parameters from previous loops
        if self.num_runs == 1:
            # More than one optimizer
            if isinstance(opt_func, list):
                self.params = self.get_module_parameters(modules)
                # Defining each of the optimizers
                self.optimizer = [
                    optimizer(module, lr=LR, weight_decay=wd)
                    for optimizer, module, LR, wd in zip(
                        opt_func, self.params, lr, weight_decay
                    )
                ]
                # Defining each of the schedulers
                if isinstance(lr_sched, (list, tuple, set)):
                    if len(lr_sched) == 1:
                        self.scheduler = [
                            lr_sched[0](
                                optimizer,
                                epoch=epochs,
                                step_per_epoch=len(train_loader),
                            )
                            for optimizer in self.optimizer
                        ]
                    else:
                        self.scheduler = [
                            (
                                scheduler(
                                    optimizer,
                                    epoch=epochs,
                                    steps_per_epoch=len(train_loader),
                                )
                                if scheduler is not None
                                else None
                            )
                            for scheduler, optimizer in zip(lr_sched, self.optimizer)
                        ]
                else:
                    if lr_sched is None:
                        self.scheduler = [lr_sched] * len(self.optimizer)
                    else:
                        self.scheduler = [
                            lr_sched(
                                optimizer,
                                epoch=epochs,
                                step_per_epoch=len(train_loader),
                            )
                            for optimizer in self.optimizer
                        ]
                # Since first run, we load the configuration of prior trainings
                self.load_config(epochs, train_loader)
            # If not more than one optimizer but different learning rates
            elif isinstance(lr, (list, tuple, set)):
                self.params = self.get_module_parameters(modules)
                # Defining just one optimizer with param groups
                self.optimizer = opt_func(
                    self.get_module_parameters(
                        modules,
                        [
                            {"lr": lr_inst, "weight_decay": wd}
                            for lr_inst, wd in zip(lr, weight_decay)
                        ],
                    )
                )

                if lr_sched is not None:
                    self.scheduler = [
                        lr_sched(
                            self.optimizer,
                            epochs=epochs,
                            steps_per_epoch=len(train_loader),
                        )
                    ]

                self.optimizer = [self.optimizer]

                self.load_config(epochs, train_loader)
            # If none of that happens, normal training loop
            else:
                self.params = self.parameters()
                # Defining the optimizer
                self.optimizer = opt_func(self.params, lr, weight_decay=weight_decay)
                # Defining the scheduler
                if lr_sched is not None:
                    self.scheduler = [
                        lr_sched(
                            self.optimizer,
                            lr,
                            epochs=epochs,
                            steps_per_epoch=len(train_loader),
                        )
                    ]
                # Putting into list for sintax
                self.optimizer = [self.optimizer]
                # Loading last config
                self.load_config(epochs, train_loader)

        if isinstance(lr, (float, int)):
            lr = [lr]
        if isinstance(weight_decay, (float, int)):
            weight_decay = [weight_decay]
        if isinstance(grad_clip, (float, int)):
            grad_clip = [grad_clip]

        if len(self.optimizer) > 1:
            for optimizer, LR, wd in zip(self.optimizer, lr, weight_decay):
                for parameter in optimizer.param_groups:
                    parameter["lr"] = LR
                    parameter["weight_decay"] = wd
        else:
            for optimizer in self.optimizer:
                for parameter, LR, wd in zip(optimizer.param_groups, lr, weight_decay):
                    parameter["lr"] = LR
                    parameter["weight_decay"] = wd
        # Clean the GPU cache
        if graph:
            assert (
                sample_input is not None
            ), "If you want to visualize a graph, you must pass through a sample tensor"
        # Build graph in tensorboard
        if graph and sample_input:
            self.writer.add_graph(self, sample_input)

        # Defining hyperparameters as attributes of the model and training object
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        for epoch in range(self.config["epoch"], self.config["epoch"] + epochs):
            # Define epoch
            self.config["epoch"] = epoch
            # Training loop
            self.train()
            for train_batch in tqdm(train_loader, desc=f"Training - Epoch: {epoch}"):
                # training step
                self.training_step(train_batch)
            # Validation loop
            self.eval()
            for val_batch in tqdm(val_loader, desc=f"Validation - Epoch: {epoch}"):
                self.validation_step(val_batch)
            # Save model and config if epoch mod(saving_div) = 0
            if epoch % saving_div == 0:
                self.save_config()
            # End of epoch
            self.end_of_epoch()
