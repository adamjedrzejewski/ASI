import optuna

# (x - 2) ** 2

def objective(trial: optuna.Trial) -> list[float]:
    x = trial.suggest_float(name="x", low=-10, high=10)
    return [(x - 2) ** 2, x]

# optuna.study.StudyDirection.MINIMIZE
# optuna.study.StudyDirection.MAXIMIZE
study = optuna.create_study(
        study_name="my-test-study",
        directions=["minimize", "maximize"],
        storage="sqlite:///trials.db",
        sampler=optuna.samplers.NSGAIISampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        load_if_exists=True
)

study.optimize(n_trials=100, n_jobs=1, func=objective, show_progress_bar=True)

