import pandas as pd
from src.training.hyperparameter_optimization import HyperparameterOptimizer

# Ajusta si tu archivo o columna target tienen otro nombre/ruta
df = pd.read_csv("data/train.csv")
target_col = "target"

X = df.drop(columns=[target_col])
y = df[target_col]

opt = HyperparameterOptimizer(X, y, n_classes=len(y.unique()), cv_folds=3)
study = opt.optimize_lightgbm_light(n_trials=5, timeout=600, sample_frac=0.25)
print("Best value:", study.best_value if hasattr(study, "best_value") else None)
print("Best params:", study.best_trial.params if hasattr(study, "best_trial") and study.best_trial is not None else None)
