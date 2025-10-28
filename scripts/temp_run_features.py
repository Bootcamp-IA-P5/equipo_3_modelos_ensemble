import pandas as pd
from src.training.hyperparameter_optimization import HyperparameterOptimizer

df = pd.read_csv("data/features_small.csv")
X = df.drop(columns=["target"])
y = df["target"]
opt = HyperparameterOptimizer(X.values, y.values, n_classes=len(y.unique()), cv_folds=3)
study = opt.optimize_lightgbm_light(n_trials=5, timeout=600, sample_frac=1.0, n_splits=3)
print("Best value:", getattr(study, "best_value", None))
print("Best params:", getattr(study.best_trial, "params", None) if getattr(study, "best_trial", None) else None)