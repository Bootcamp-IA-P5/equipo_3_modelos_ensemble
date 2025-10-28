"""
Script principal de entrenamiento completo.
Entrena CNN para extraer embeddings y modelos ensemble.
"""

import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
import argparse

from src.data.preprocessing import DataPreprocessor
from src.models.cnn_extractor import get_model, EmbeddingExtractor
from src.models.ensemble_models import EnsembleModels
from src.training.cnn_trainer import CNNTrainer
from src.training.hyperparameter_optimization import HyperparameterOptimizer, CrossValidator


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenar modelos de clasificación de residuos')
    
    parser.add_argument('--data_dir', type=str, 
                       default='data/raw/garbage_classification',
                       help='Directorio con datos')
    
    parser.add_argument('--img_size', type=int, default=128,
                       help='Tamaño de imagen (default: 128 para 6GB VRAM)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Número de épocas para CNN')
    
    parser.add_argument('--model_name', type=str, default='resnet18',
                       choices=['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
                       help='Modelo CNN base')
    
    parser.add_argument('--optimize_hp', action='store_true',
                       help='Optimizar hiperparámetros de ensemble')
    
    parser.add_argument('--hp_trials', type=int, default=30,
                       help='Número de trials para optimización')
    
    parser.add_argument('--output_dir', type=str, default='models/trained',
                       help='Directorio de salida')
    
    parser.add_argument('--skip_cnn', action='store_true',
                       help='Saltar entrenamiento de CNN (usar embeddings guardados)')
    
    return parser.parse_args()


def main():
    """Función principal de entrenamiento."""
    args = parse_args()
    
    print("="*80)
    print("🗑️  ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN DE RESIDUOS")
    print("="*80)
    print(f"\n⚙️  Configuración:")
    print(f"   • Directorio de datos: {args.data_dir}")
    print(f"   • Tamaño de imagen: {args.img_size}x{args.img_size}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Modelo CNN: {args.model_name}")
    print(f"   • Épocas: {args.epochs}")
    print(f"   • Optimizar HP: {args.optimize_hp}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   • Dispositivo: {device}")
    
    if device == 'cuda':
        print(f"   • GPU: {torch.cuda.get_device_name(0)}")
        print(f"   • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"exp_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Directorio de experimento: {experiment_dir}")
    
    # ====================
    # 1. PREPROCESAMIENTO
    # ====================
    print("\n" + "="*80)
    print("1️⃣  PREPROCESAMIENTO DE DATOS")
    print("="*80)
    
    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=42
    )
    
    # Crear splits
    splits_path = Path('data/processed/splits.json')
    
    if not splits_path.exists():
        splits = preprocessor.create_splits(val_size=0.15, test_size=0.15)
        preprocessor.save_splits(splits, splits_path)
    else:
        print(f"\n✅ Cargando splits existentes...")
        splits = preprocessor.load_splits(splits_path)
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(splits, num_workers=4)
    
    # ====================
    # 2. ENTRENAR CNN
    # ====================
    if not args.skip_cnn:
        print("\n" + "="*80)
        print("2️⃣  ENTRENAMIENTO DE CNN (Transfer Learning)")
        print("="*80)
        
        # Crear modelo
        cnn_model = get_model(
            model_name=args.model_name,
            num_classes=6,
            pretrained=True
        )
        
        # Calcular class weights
        train_labels = np.array(splits['train']['labels'])
        class_counts = np.bincount(train_labels)
        
        print(f"\n📊 Distribución de clases en train:")
        for idx, count in enumerate(class_counts):
            class_name = preprocessor.CLASSES[idx]
            print(f"   • {class_name}: {count}")
        
        # Crear trainer
        trainer = CNNTrainer(
            model=cnn_model,
            device=device,
            learning_rate=0.001,
            weight_decay=1e-4
        )
        
        # Establecer class weights
        trainer.set_class_weights(class_counts)
        
        # Entrenar
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            early_stopping_patience=10,
            save_dir=experiment_dir
        )
        
        # Guardar historia
        trainer.save_history(experiment_dir / 'cnn_training_history.json')
        
        # Plotear historia
        trainer.plot_training_history(experiment_dir / 'cnn_training_curves.png')
        
        # Guardar modelo final
        torch.save(cnn_model.state_dict(), experiment_dir / 'cnn_model_final.pth')
        
        print(f"\n✅ CNN entrenada y guardada")
        
    else:
        print("\n⏭️  Saltando entrenamiento de CNN")
        # Cargar modelo guardado
        cnn_model = get_model(args.model_name, num_classes=6, pretrained=False)
        model_path = experiment_dir / 'best_model.pth'
        if model_path.exists():
            checkpoint = torch.load(model_path)
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Modelo CNN cargado desde: {model_path}")
        else:
            print(f"⚠️  No se encontró modelo guardado en {model_path}")
            print(f"   Se usará modelo pre-entrenado sin fine-tuning")
    
    # ====================
    # 3. EXTRAER EMBEDDINGS
    # ====================
    print("\n" + "="*80)
    print("3️⃣  EXTRACCIÓN DE EMBEDDINGS")
    print("="*80)
    
    extractor = EmbeddingExtractor(model=cnn_model, device=device)
    
    # Extraer embeddings de cada conjunto
    print(f"\n📤 Extrayendo embeddings de train...")
    X_train_emb, y_train = extractor.extract_from_dataloader(train_loader)
    
    print(f"\n📤 Extrayendo embeddings de validation...")
    X_val_emb, y_val = extractor.extract_from_dataloader(val_loader)
    
    print(f"\n📤 Extrayendo embeddings de test...")
    X_test_emb, y_test = extractor.extract_from_dataloader(test_loader)
    
    # Guardar embeddings
    np.savez_compressed(
        experiment_dir / 'embeddings.npz',
        X_train=X_train_emb,
        y_train=y_train,
        X_val=X_val_emb,
        y_val=y_val,
        X_test=X_test_emb,
        y_test=y_test
    )
    
    print(f"\n✅ Embeddings guardados")
    print(f"   • Train shape: {X_train_emb.shape}")
    print(f"   • Val shape: {X_val_emb.shape}")
    print(f"   • Test shape: {X_test_emb.shape}")
    
    # ====================
    # 4. OPTIMIZAR HIPERPARÁMETROS (OPCIONAL)
    # ====================
    best_params = {}
    
    if args.optimize_hp:
        print("\n" + "="*80)
        print("4️⃣  OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("="*80)
        
        optimizer = HyperparameterOptimizer(
            X_train=X_train_emb,
            y_train=y_train,
            n_classes=6,
            cv_folds=5,
            random_state=42
        )
        
        # Optimizar cada modelo
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            print(f"\n{'─'*80}")
            best_params[model_type] = optimizer.optimize(
                model_type=model_type,
                n_trials=args.hp_trials
            )
        
        # Guardar mejores parámetros
        optimizer.save_best_params(experiment_dir / 'best_hyperparameters.json')
    
    # ====================
    # 5. ENTRENAR MODELOS ENSEMBLE
    # ====================
    print("\n" + "="*80)
    print("5️⃣  ENTRENAMIENTO DE MODELOS ENSEMBLE")
    print("="*80)
    
    ensemble = EnsembleModels(n_classes=6, random_state=42)
    
    # Lista de modelos a entrenar
    models_to_train = [
        ('random_forest', ensemble.create_random_forest()),
        ('xgboost', ensemble.create_xgboost()),
        ('lightgbm', ensemble.create_lightgbm()),
        ('voting', ensemble.create_voting_ensemble()),
        ('stacking', ensemble.create_stacking_ensemble())
    ]
    
    # Entrenar cada modelo
    results = {}
    
    for model_name, model in models_to_train:
        print(f"\n{'─'*80}")
        
        # Entrenar
        train_results = ensemble.train_model(
            model=model,
            X_train=X_train_emb,
            y_train=y_train,
            model_name=model_name
        )
        
        # Evaluar en validation
        val_results = ensemble.evaluate_model(
            model_name=model_name,
            X_test=X_val_emb,
            y_test=y_val,
            class_names=preprocessor.CLASSES
        )
        
        # Guardar modelo
        ensemble.save_model(
            model_name=model_name,
            save_path=experiment_dir / f'{model_name}_model.pkl'
        )
        
        # Guardar resultados
        results[model_name] = {
            'train_accuracy': train_results['train_accuracy'],
            'val_accuracy': val_results['test_accuracy'],
            'overfitting': train_results['train_accuracy'] - val_results['test_accuracy']
        }
        
        print(f"   • Val Accuracy: {val_results['test_accuracy']:.4f}")
        print(f"   • Overfitting: {results[model_name]['overfitting']*100:.2f}%")
    
    # ====================
    # 6. EVALUACIÓN FINAL EN TEST
    # ====================
    print("\n" + "="*80)
    print("6️⃣  EVALUACIÓN FINAL EN TEST SET")
    print("="*80)
    
    final_results = {}
    
    for model_name in ensemble.models.keys():
        print(f"\n{'─'*80}")
        print(f"📊 Evaluando {model_name} en test set...")
        
        test_results = ensemble.evaluate_model(
            model_name=model_name,
            X_test=X_test_emb,
            y_test=y_test,
            class_names=preprocessor.CLASSES
        )
        
        final_results[model_name] = {
            'train_accuracy': results[model_name]['train_accuracy'],
            'val_accuracy': results[model_name]['val_accuracy'],
            'test_accuracy': test_results['test_accuracy'],
            'overfitting': results[model_name]['overfitting'],
            'classification_report': test_results['classification_report'],
            'confusion_matrix': test_results['confusion_matrix'].tolist()
        }
        
        print(f"   • Test Accuracy: {test_results['test_accuracy']:.4f}")
    
    # Guardar resultados finales
    with open(experiment_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # ====================
    # 7. RESUMEN
    # ====================
    print("\n" + "="*80)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*80)
    
    print(f"\n{'Modelo':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Overfitting':<12}")
    print("─"*80)
    
    for model_name, res in final_results.items():
        print(f"{model_name:<20} "
              f"{res['train_accuracy']*100:>10.2f}%  "
              f"{res['val_accuracy']*100:>10.2f}%  "
              f"{res['test_accuracy']*100:>10.2f}%  "
              f"{res['overfitting']*100:>10.2f}%")
    
    # Mejor modelo
    best_model = max(final_results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n🏆 Mejor modelo: {best_model[0]} (Test Acc: {best_model[1]['test_accuracy']*100:.2f}%)")
    
    # Verificar overfitting
    print(f"\n⚠️  Modelos con overfitting > 5%:")
    for model_name, res in final_results.items():
        if res['overfitting'] > 0.05:
            print(f"   • {model_name}: {res['overfitting']*100:.2f}%")
    
    print(f"\n✅ Entrenamiento completado")
    print(f"📁 Resultados guardados en: {experiment_dir}")
    
    # Guardar config del experimento
    config = {
        'timestamp': timestamp,
        'args': vars(args),
        'device': device,
        'results': final_results
    }
    
    with open(experiment_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
