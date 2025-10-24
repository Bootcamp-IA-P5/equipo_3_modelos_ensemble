"""
Módulo de preprocesamiento de datos para clasificación de residuos.
Optimizado para hardware con 6GB VRAM y 32GB RAM.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm


class GarbageDataset(Dataset):
    """Dataset personalizado para carga eficiente de imágenes de residuos."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: Lista de rutas a imágenes
            labels: Lista de etiquetas (índices de clase)
            transform: Transformaciones de torchvision
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


class DataPreprocessor:
    """Clase para preprocesar y crear splits del dataset de residuos."""
    
    # Clases del dataset (en orden alfabético para consistencia)
    CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    def __init__(self, 
                 data_dir: str,
                 img_size: int = 128,  # Reducido para 6GB VRAM
                 batch_size: int = 32,
                 seed: int = 42):
        """
        Args:
            data_dir: Directorio raíz con subdirectorios por clase
            img_size: Tamaño de imagen (128x128 para optimización de memoria)
            batch_size: Tamaño de batch
            seed: Semilla para reproducibilidad
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        
        # Mapeos clase <-> índice
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"📁 Directorio de datos: {self.data_dir}")
        print(f"🏷️  Clases: {self.CLASSES}")
        print(f"📐 Tamaño de imagen: {self.img_size}x{self.img_size}")
        
    def get_transforms(self, mode: str = 'train') -> transforms.Compose:
        """
        Obtiene las transformaciones según el modo.
        
        Args:
            mode: 'train', 'val' o 'test'
            
        Returns:
            Transformaciones de torchvision
        """
        if mode == 'train':
            # Data augmentation para entrenamiento
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet stats
            ])
        else:
            # Sin augmentation para validación y test
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def load_image_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        """
        Carga las rutas de imágenes y sus etiquetas.
        
        Returns:
            Tupla de (rutas_imagenes, etiquetas)
        """
        image_paths = []
        labels = []
        
        print("\n📂 Cargando rutas de imágenes...")
        
        for class_name in self.CLASSES:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️  Advertencia: No se encontró el directorio {class_dir}")
                continue
            
            # Obtener todas las imágenes de la clase
            class_images = list(class_dir.glob('*.jpg')) + \
                          list(class_dir.glob('*.jpeg')) + \
                          list(class_dir.glob('*.png'))
            
            class_label = self.class_to_idx[class_name]
            
            for img_path in class_images:
                image_paths.append(str(img_path))
                labels.append(class_label)
            
            print(f"  ✓ {class_name}: {len(class_images)} imágenes")
        
        print(f"\n✅ Total: {len(image_paths)} imágenes cargadas")
        
        return image_paths, labels
    
    def create_splits(self, 
                     val_size: float = 0.15,
                     test_size: float = 0.15) -> Dict:
        """
        Crea splits estratificados de train/val/test.
        
        Args:
            val_size: Proporción para validación
            test_size: Proporción para test
            
        Returns:
            Diccionario con los splits
        """
        # Cargar datos
        image_paths, labels = self.load_image_paths_and_labels()
        
        # Convertir a numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Calcular proporciones
        train_size = 1.0 - val_size - test_size
        test_size_adjusted = test_size / (val_size + test_size)
        
        print(f"\n📊 Creando splits:")
        print(f"  • Train: {train_size*100:.1f}%")
        print(f"  • Validation: {val_size*100:.1f}%")
        print(f"  • Test: {test_size*100:.1f}%")
        
        # Primer split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels,
            test_size=(val_size + test_size),
            stratify=labels,
            random_state=self.seed
        )
        
        # Segundo split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size_adjusted,
            stratify=y_temp,
            random_state=self.seed
        )
        
        # Mostrar distribución
        print("\n📈 Distribución por clase:")
        print("="*60)
        print(f"{'Clase':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        print("="*60)
        
        for class_name, class_idx in self.class_to_idx.items():
            train_count = np.sum(y_train == class_idx)
            val_count = np.sum(y_val == class_idx)
            test_count = np.sum(y_test == class_idx)
            total_count = train_count + val_count + test_count
            
            print(f"{class_name:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")
        
        print("="*60)
        print(f"{'TOTAL':<15} {len(y_train):<10} {len(y_val):<10} {len(y_test):<10} {len(labels):<10}")
        print("="*60)
        
        splits = {
            'train': {'paths': X_train.tolist(), 'labels': y_train.tolist()},
            'val': {'paths': X_val.tolist(), 'labels': y_val.tolist()},
            'test': {'paths': X_test.tolist(), 'labels': y_test.tolist()},
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'classes': self.CLASSES
        }
        
        return splits
    
    def save_splits(self, splits: Dict, output_path: str):
        """Guarda los splits en archivo JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"\n✅ Splits guardados en: {output_path}")
    
    def load_splits(self, splits_path: str) -> Dict:
        """Carga splits desde archivo JSON."""
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        print(f"✅ Splits cargados desde: {splits_path}")
        return splits
    
    def get_dataloaders(self, 
                       splits: Dict,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Crea DataLoaders para train, val y test.
        
        Args:
            splits: Diccionario con los splits
            num_workers: Número de workers para carga paralela
            
        Returns:
            Tupla de (train_loader, val_loader, test_loader)
        """
        # Crear datasets
        train_dataset = GarbageDataset(
            splits['train']['paths'],
            splits['train']['labels'],
            transform=self.get_transforms('train')
        )
        
        val_dataset = GarbageDataset(
            splits['val']['paths'],
            splits['val']['labels'],
            transform=self.get_transforms('val')
        )
        
        test_dataset = GarbageDataset(
            splits['test']['paths'],
            splits['test']['labels'],
            transform=self.get_transforms('test')
        )
        
        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"\n✅ DataLoaders creados:")
        print(f"  • Train batches: {len(train_loader)}")
        print(f"  • Val batches: {len(val_loader)}")
        print(f"  • Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader


def main():
    """Función principal para probar el preprocesamiento."""
    # Configuración
    DATA_DIR = Path('../data/raw/garbage_classification')
    OUTPUT_DIR = Path('../data/processed')
    
    # Crear preprocessor
    preprocessor = DataPreprocessor(
        data_dir=DATA_DIR,
        img_size=128,
        batch_size=32,
        seed=42
    )
    
    # Crear y guardar splits
    splits = preprocessor.create_splits(val_size=0.15, test_size=0.15)
    preprocessor.save_splits(splits, OUTPUT_DIR / 'splits.json')
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(splits)
    
    # Probar un batch
    print("\n🧪 Probando carga de un batch...")
    images, labels = next(iter(train_loader))
    print(f"  • Shape de imágenes: {images.shape}")
    print(f"  • Shape de etiquetas: {labels.shape}")
    print(f"  • Rango de valores: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\n✅ ¡Preprocesamiento completado exitosamente!")


if __name__ == '__main__':
    main()
