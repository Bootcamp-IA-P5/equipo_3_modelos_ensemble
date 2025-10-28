"""
Modelos de ensemble para clasificación basados en embeddings.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
from typing import Dict, Tuple
import json


class EnsembleModels:
    """Colección de modelos ensemble para clasificación."""
    
    def __init__(self, n_classes: int = 6, random_state: int = 42):
        """
        Args:
            n_classes: Número de clases
            random_state: Semilla para reproducibilidad
        """
        self.n_classes = n_classes
        self.random_state = random_state
        self.models = {}
        
        print(f"🎯 Inicializando modelos ensemble...")
        print(f"   • Número de clases: {n_classes}")
    
    def create_random_forest(self, n_estimators: int = 200) -> RandomForestClassifier:
        """
        Crea modelo Random Forest.
        
        Args:
            n_estimators: Número de árboles
            
        Returns:
            Random Forest classifier
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',  # Para manejar desbalanceo
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        print(f"✅ Random Forest creado ({n_estimators} árboles)")
        return model
    
    def create_xgboost(self) -> XGBClassifier:
        """
        Crea modelo XGBoost.
        
        Returns:
            XGBoost classifier
        """
        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=self.n_classes,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        print(f"✅ XGBoost creado")
        return model
    
    def create_lightgbm(self) -> LGBMClassifier:
        """
        Crea modelo LightGBM.
        
        Returns:
            LightGBM classifier
        """
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        print(f"✅ LightGBM creado")
        return model
    
    def create_svm(self) -> SVC:
        """
        Crea modelo SVM.
        
        Returns:
            SVM classifier
        """
        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=self.random_state,
            verbose=False
        )
        
        print(f"✅ SVM creado")
        return model
    
    def create_voting_ensemble(self) -> VotingClassifier:
        """
        Crea ensemble por votación.
        
        Returns:
            Voting classifier
        """
        # Modelos base
        rf = self.create_random_forest(n_estimators=100)
        xgb = self.create_xgboost()
        lgbm = self.create_lightgbm()
        
        # Voting classifier (soft voting con probabilidades)
        voting = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb),
                ('lgbm', lgbm)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        print(f"✅ Voting Ensemble creado (RF + XGBoost + LightGBM)")
        return voting
    
    def create_stacking_ensemble(self) -> StackingClassifier:
        """
        Crea ensemble por stacking.
        
        Returns:
            Stacking classifier
        """
        # Modelos base
        rf = self.create_random_forest(n_estimators=100)
        xgb = self.create_xgboost()
        lgbm = self.create_lightgbm()
        
        # Meta-learner: Regresión Logística
        meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Stacking classifier
        stacking = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb),
                ('lgbm', lgbm)
            ],
            final_estimator=meta_learner,
            cv=5,  # Cross-validation folds para stacking
            n_jobs=-1
        )
        
        print(f"✅ Stacking Ensemble creado (RF + XGBoost + LightGBM -> LogReg)")
        return stacking
    
    def train_model(self, 
                   model, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   model_name: str) -> Dict:
        """
        Entrena un modelo.
        
        Args:
            model: Modelo a entrenar
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        print(f"\n🏋️  Entrenando {model_name}...")
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # Predicciones en train
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        print(f"✅ {model_name} entrenado")
        print(f"   • Train Accuracy: {train_acc:.4f}")
        
        # Guardar modelo
        self.models[model_name] = model
        
        return {
            'model_name': model_name,
            'train_accuracy': train_acc
        }
    
    def evaluate_model(self,
                      model_name: str,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      class_names: list) -> Dict:
        """
        Evalúa un modelo.
        
        Args:
            model_name: Nombre del modelo
            X_test: Features de test
            y_test: Labels de test
            class_names: Nombres de las clases
            
        Returns:
            Diccionario con métricas
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        model = self.models[model_name]
        
        print(f"\n📊 Evaluando {model_name}...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Métricas
        test_acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        print(f"✅ Test Accuracy: {test_acc:.4f}")
        
        return {
            'model_name': model_name,
            'test_accuracy': test_acc,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
    
    def save_model(self, model_name: str, save_path: str):
        """
        Guarda un modelo entrenado.
        
        Args:
            model_name: Nombre del modelo
            save_path: Ruta donde guardar
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.models[model_name], save_path)
        
        print(f"✅ Modelo '{model_name}' guardado en: {save_path}")
    
    def load_model(self, model_name: str, load_path: str):
        """
        Carga un modelo guardado.
        
        Args:
            model_name: Nombre para el modelo
            load_path: Ruta del modelo guardado
        """
        self.models[model_name] = joblib.load(load_path)
        
        print(f"✅ Modelo '{model_name}' cargado desde: {load_path}")
    
    def get_feature_importance(self, model_name: str, feature_names: list = None) -> np.ndarray:
        """
        Obtiene feature importance si el modelo lo soporta.
        
        Args:
            model_name: Nombre del modelo
            feature_names: Nombres de las features
            
        Returns:
            Array con importancias
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        model = self.models[model_name]
        
        # Obtener importancias según el tipo de modelo
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print(f"⚠️  Modelo '{model_name}' no soporta feature importance")
            return None
        
        if feature_names is not None and len(feature_names) == len(importances):
            # Ordenar por importancia
            indices = np.argsort(importances)[::-1]
            
            print(f"\n📊 Top 10 Features más importantes ({model_name}):")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return importances


def main():
    """Test de los modelos ensemble."""
    print("🧪 Probando modelos ensemble...\n")
    
    # Datos sintéticos para prueba
    np.random.seed(42)
    X_train = np.random.randn(100, 512)  # 100 muestras, 512 features (embeddings)
    y_train = np.random.randint(0, 6, 100)
    X_test = np.random.randn(30, 512)
    y_test = np.random.randint(0, 6, 30)
    
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Crear ensemble
    ensemble = EnsembleModels(n_classes=6)
    
    # Entrenar Random Forest
    rf_model = ensemble.create_random_forest()
    ensemble.train_model(rf_model, X_train, y_train, 'random_forest')
    
    # Evaluar
    results = ensemble.evaluate_model('random_forest', X_test, y_test, class_names)
    
    print(f"\n✅ Test completado")
    print(f"   • Test Accuracy: {results['test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
