"""
Optimización de hiperparámetros usando Optuna.

Contiene:
- HyperparameterOptimizer: clase principal con métodos para RandomForest, XGBoost, LightGBM.
- Método adicional optimize_lightgbm_light para tuning ligero (sampling, pruning, sqlite persistence).
"""

import json
import os
from pathlib import Path
from typing import Dict
import lightgbm as lgb
import numpy as np
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class HyperparameterOptimizer:
    """Optimizador de hiperparámetros usando Optuna."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: int = 6,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            n_classes: Número de clases
            cv_folds: Número de folds para cross-validation
            random_state: Semilla
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_classes = n_classes
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        self.best_params = {}

        print("🔍 Optimizador inicializado")
        print(f"   • Samples: {len(X_train)}")
        print(f"   • Features: {X_train.shape[1]}")
        print(f"   • CV Folds: {cv_folds}")

    # ---------- Objetivos existentes (mantengo tus originales) ----------
    def objective_random_forest(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 30, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        model = RandomForestClassifier(**params)

        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy", n_jobs=-1)

        return float(scores.mean())

    def objective_xgboost(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            "objective": "multi:softmax",
            "num_class": self.n_classes,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = XGBClassifier(**params)

        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy", n_jobs=-1)

        return float(scores.mean())

    def objective_lightgbm(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }

        model = LGBMClassifier(**params)

        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy", n_jobs=-1)

        return float(scores.mean())

    # ---------- Optimización general (mantengo tu flujo) ----------
    def optimize(self, model_type: str, n_trials: int = 50, timeout: int = None) -> Dict:
        """
        Optimiza hiperparámetros.

        Args:
            model_type: 'random_forest', 'xgboost', 'lightgbm'
            n_trials: Número de trials
            timeout: Timeout en segundos

        Returns:
            Diccionario con mejores parámetros y score
        """
        print(f"\n🔍 Optimizando {model_type}...")
        print(f"   • Trials: {n_trials}")

        if model_type == "random_forest":
            objective = self.objective_random_forest
        elif model_type == "xgboost":
            objective = self.objective_xgboost
        elif model_type == "lightgbm":
            objective = self.objective_lightgbm
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\n✅ Optimización completada")
        print(f"   • Mejor CV Score: {best_score:.4f}")
        print("   • Mejores parámetros:")
        for param, value in best_params.items():
            print(f"     - {param}: {value}")

        self.best_params[model_type] = {"params": best_params, "cv_score": best_score, "n_trials": len(study.trials)}

        return self.best_params[model_type]

    # ---------- NUEVO: objective y optimize ligeros para LightGBM ----------
    def _save_optuna_best(self, study: optuna.Study, path: str = "optuna_best_params.json"):
        """Guarda el mejor trial del estudio en JSON y actualiza self.best_params['lightgbm_light']"""
        try:
            if study.best_trial is not None:
                payload = {"value": float(study.best_value), "params": study.best_trial.params}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                # Actualiza self.best_params para coherencia con el resto de la clase
                self.best_params["lightgbm_light"] = {
                    "params": study.best_trial.params,
                    "cv_score": float(study.best_value),
                    "n_trials": len(study.trials),
                }
        except Exception:
            # No queremos que errores de IO paren el proceso de tuning
            pass

    def objective_lightgbm_light(self, trial: optuna.Trial, sample_frac: float = 0.3, n_splits: int = 3) -> float:
        """
        Objective ligero para LightGBM:
        - Usa una fracción del dataset para tuning rápido (sample_frac).
        - CV reducido (n_splits, típico 3) para menor coste.
        - early_stopping + LightGBMPruningCallback.
        Devuelve F1 macro promedio.
        """
        # Sample para acelerar la evaluación durante el tuning
        if 0 < sample_frac < 1.0:
            Xs = self._pandasify_X(self.X_train).sample(frac=sample_frac, random_state=self.random_state)
            ys = self._pandasify_y(self.y_train).loc[Xs.index]
        else:
            Xs = self._pandasify_X(self.X_train)
            ys = self._pandasify_y(self.y_train)

        params = {
            "objective": "multiclass",
            "num_class": int(self.n_classes),
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 2e-1),
            "subsample": trial.suggest_uniform("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 10.0),
            "random_state": self.random_state,
            "n_jobs": 1,  # evitar nested parallelism dentro de trials
            "verbosity": -1,
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(Xs, ys)):
            X_train_fold = Xs.iloc[train_idx]
            X_val_fold = Xs.iloc[val_idx]
            y_train_fold = ys.iloc[train_idx]
            y_val_fold = ys.iloc[val_idx]

            clf = LGBMClassifier(**params)
            clf.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=30),],
            )

            y_pred_proba = clf.predict_proba(X_val_fold, num_iteration=clf.best_iteration_)
            y_pred = np.argmax(y_pred_proba, axis=1)
            fold_scores.append(f1_score(y_val_fold, y_pred, average="macro"))

            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    def optimize_lightgbm_light(
        self,
        study_name: str = "lightgbm_light_tuning",
        storage_path: str = "sqlite:///optuna_lightgbm.db",
        n_trials: int = 20,
        timeout: int = None,
        sample_frac: float = 0.3,
        n_splits: int = 3,
        save_path: str = "optuna_best_params.json",
        direction: str = "maximize",
    ):
        """
        Ejecuta un tuning ligero para LightGBM con Optuna:
        - Guarda el estudio en SQLite para reanudar.
        - Usa MedianPruner para cortar trials mediocres.
        - Guarda best params en JSON tras cada trial (callback).
        """
        # Asegurar carpeta para sqlite si es ruta local
        if storage_path.startswith("sqlite:///"):
            db_file = storage_path.replace("sqlite:///", "", 1)
            db_dir = os.path.dirname(db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        pruner = MedianPruner(n_startup_trials=5)
        study = optuna.create_study(
            study_name=study_name, storage=storage_path, load_if_exists=True, direction=direction, pruner=pruner
        )

        def _cb(s, t):
            self._save_optuna_best(s, path=save_path)

        try:
            study.optimize(
                lambda tr: self.objective_lightgbm_light(tr, sample_frac=sample_frac, n_splits=n_splits),
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[_cb],
            )
        except KeyboardInterrupt:
            print("Optimización interrumpida: guardando mejores parámetros...")
            self._save_optuna_best(study, path=save_path)
        except Exception:
            # Guardar en caso de fallo inesperado
            self._save_optuna_best(study, path=save_path)
            raise
        finally:
            self._save_optuna_best(study, path=save_path)

        return study

    # ---------- utilidades ----------
    def save_best_params(self, save_path: str):
        """
        Guarda mejores parámetros (diccionario self.best_params) en JSON.

        Args:
            save_path: Ruta del archivo JSON
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=2)

        print(f"✅ Parámetros guardados en: {save_path}")

    def load_best_params(self, load_path: str):
        """
        Carga mejores parámetros a self.best_params desde JSON.

        Args:
            load_path: Ruta del archivo JSON
        """
        with open(load_path, "r", encoding="utf-8") as f:
            self.best_params = json.load(f)

        print(f"✅ Parámetros cargados desde: {load_path}")

    # Helpers para aceptar tanto numpy arrays como DataFrame/Series
    def _pandasify_X(self, X):
        """Devuelve pandas DataFrame si X es numpy array; si ya es DataFrame lo devuelve."""
        try:
            import pandas as pd

            if isinstance(X, (pd.DataFrame,)):
                return X
            return pd.DataFrame(X)
        except Exception:
            # Si pandas no está disponible, asumimos que X ya tiene el índice/iloc compatible
            return X

    def _pandasify_y(self, y):
        """Devuelve pandas Series si y es numpy array; si ya es Series lo devuelve."""
        try:
            import pandas as pd

            if isinstance(y, (pd.Series,)):
                return y
            return pd.Series(y)
        except Exception:
            return y


if __name__ == "__main__":
    # Test rápido de humo
    print("🧪 Probando optimizador...\n")

    # Datos sintéticos (ejemplo)
    np.random.seed(42)
    X = np.random.randn(200, 512)
    y = np.random.randint(0, 6, 200)

    optimizer = HyperparameterOptimizer(X, y, n_classes=6, cv_folds=3)
    # prueba del optimizador ligero (muy pocos trials para humo)
    study = optimizer.optimize_lightgbm_light(n_trials=3, timeout=300, sample_frac=0.2, n_splits=3)
    print("Best value (light):", study.best_value if study.best_trial is not None else None)
    print("✅ Test completado")