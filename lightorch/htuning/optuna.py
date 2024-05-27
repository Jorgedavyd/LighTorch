import optuna
import torch

from lightning.pytorch import LightningDataModule, LightningModule
from typing import Sequence, Callable, Dict, Any
from lightning.pytorch.trainer import Trainer


def htuning(
    *,
    model_class: LightningModule,
    hparam_objective: Callable[[optuna.trial.Trial], Sequence[float]],
    datamodule: LightningDataModule,
    valid_metrics: Sequence[str],
    datamodule_kwargs: Dict[str, Any] = dict(
        pin_memory=True, num_workers=8, batch_size=6
    ),
    directions: Sequence[str] | str,
    precision: str,
    n_trials: int,
    trainer_kwargs: Dict[str, Any] = dict(
        logger=True,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="cuda",
        devices=1,
        log_every_n_steps=22,
        precision="bf16-mixed",
        limit_train_batches=1 / 3,
        limit_val_batches=1 / 3,
    ),
    **kwargs,
):

    def objective(trial: optuna.trial.Trial):

        dataset = datamodule(**datamodule_kwargs)

        model = model_class(**hparam_objective(trial))

        trainer = Trainer(**trainer_kwargs)

        trainer.fit(model, datamodule=dataset)

        if isinstance(valid_metrics, str):
            return trainer.callback_metrics[valid_metrics].item()

        return (
            trainer.callback_metrics[valid_metric].item()
            for valid_metric in valid_metrics
        )

    if "precision" in kwargs:
        torch.set_float32_matmul_precision(precision)
    else:
        torch.set_float32_matmul_precision("medium")

    if isinstance(valid_metrics, str):
        # Single objective optimization
        study = optuna.create_study(
            direction=directions,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=True,
            n_jobs=-1,
        )

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        # MultiObjective optimization
        study = optuna.create_study(directions=directions)
        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=True,
            n_jobs=-1,
        )
        for i, name in enumerate(valid_metrics):
            best_param = max(study.best_trials, key=lambda t: t.values[i])
            print(f"Trial with best {name}:")
            print(f"\tnumber: {best_param.number}")
            print(f"\tparams: {best_param.params}")
            print(f"\tvalues: {best_param.values}")
