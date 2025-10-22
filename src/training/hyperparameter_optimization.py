"""
Optimización de hiperparámetros usando Optuna.
"""

import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Callable
import json
from pathlib import Path


class HyperparameterOptimizer:
    """Optimizador de hiperparámetros usando Optuna."""
    
    def __init__(self, 
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 n_classes: int = 6,
                 cv_folds: int = 5,
                 random_state: int = 42):
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
        
        self.cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )
        
        self.best_params = {}
        
        print(f"🔍 Optimizador inicializado")
        print(f"   • Samples: {len(X_train)}")
        print(f"   • Features: {X_train.shape[1]}")
        print(f"   • CV Folds: {cv_folds}")
    
    def objective_random_forest(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para Random Forest.
        
        Args:
            trial: Trial de Optuna
            
        Returns:
            Score de cross-validation
        """
        # Hiperparámetros a optimizar
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def objective_xgboost(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para XGBoost.
        
        Args:
            trial: Trial de Optuna
            
        Returns:
            Score de cross-validation
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'objective': 'multi:softmax',
            'num_class': self.n_classes,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = XGBClassifier(**params)
        
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def objective_lightgbm(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para LightGBM.
        
        Args:
            trial: Trial de Optuna
            
        Returns:
            Score de cross-validation
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def optimize(self,
                model_type: str,
                n_trials: int = 50,
                timeout: int = None) -> Dict:
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
        
        # Seleccionar función objetivo
        if model_type == 'random_forest':
            objective = self.objective_random_forest
        elif model_type == 'xgboost':
            objective = self.objective_xgboost
        elif model_type == 'lightgbm':
            objective = self.objective_lightgbm
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Crear estudio
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Optimizar
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Resultados
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n✅ Optimización completada")
        print(f"   • Mejor CV Score: {best_score:.4f}")
        print(f"   • Mejores parámetros:")
        for param, value in best_params.items():
            print(f"     - {param}: {value}")
        
        self.best_params[model_type] = {
            'params': best_params,
            'cv_score': best_score,
            'n_trials': len(study.trials)
        }
        
        return self.best_params[model_type]
    
    def save_best_params(self, save_path: str):
        """
        Guarda mejores parámetros.
        
        Args:
            save_path: Ruta del archivo JSON
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"✅ Parámetros guardados en: {save_path}")
    
    def load_best_params(self, load_path: str):
        """
        Carga mejores parámetros.
        
        Args:
            load_path: Ruta del archivo JSON
        """
        with open(load_path, 'r') as f:
            self.best_params = json.load(f)
        
        print(f"✅ Parámetros cargados desde: {load_path}")


class CrossValidator:
    """Validación cruzada estratificada."""
    
    def __init__(self,
                 n_splits: int = 5,
                 random_state: int = 42):
        """
        Args:
            n_splits: Número de folds
            random_state: Semilla
        """
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        
        print(f"🔄 Cross-Validator creado ({n_splits} folds)")
    
    def evaluate(self,
                model,
                X: np.ndarray,
                y: np.ndarray,
                scoring: str = 'accuracy') -> Dict:
        """
        Evalúa modelo con cross-validation.
        
        Args:
            model: Modelo a evaluar
            X: Features
            y: Labels
            scoring: Métrica
            
        Returns:
            Diccionario con resultados
        """
        print(f"\n🔄 Evaluando con {self.n_splits}-Fold Cross-Validation...")
        
        scores = cross_val_score(
            model, X, y,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        results = {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        print(f"✅ Resultados:")
        print(f"   • Mean {scoring}: {results['mean']:.4f} (±{results['std']:.4f})")
        print(f"   • Min: {results['min']:.4f} | Max: {results['max']:.4f}")
        
        return results


if __name__ == '__main__':
    # Test
    print("🧪 Probando optimizador...\n")
    
    # Datos sintéticos
    np.random.seed(42)
    X = np.random.randn(200, 512)
    y = np.random.randint(0, 6, 200)
    
    # Optimizar Random Forest (solo 5 trials para test rápido)
    optimizer = HyperparameterOptimizer(X, y, n_classes=6, cv_folds=3)
    best_params = optimizer.optimize('random_forest', n_trials=5)
    
    print("\n✅ Test completado")
