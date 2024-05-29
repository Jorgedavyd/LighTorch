# Hyperparameter Tuning and Multi-Objective optimization with genetic algorithms
We can search in the hyperparameter space the Pareto Frontier for a multi-objective case, and also for the single-objective case. This is accomplished through optuna built-in algorithms and framework.
## htuning

```python 
from lightorch.htuning.optuna import htuning
from ... import DataModule
from ... import Model

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