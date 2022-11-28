import optuna

# (x - 2) ** 2

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float(name="x", low=-10, high=10)
    return (x - 2) ** 2

