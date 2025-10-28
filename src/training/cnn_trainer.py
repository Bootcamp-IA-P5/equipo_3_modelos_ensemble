"""
Script de entrenamiento para modelo CNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class CNNTrainer:
    """Entrenador para modelos CNN."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        """
        Args:
            model: Modelo a entrenar
            device: Dispositivo de cómputo
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer y scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Loss function con class weights
        self.criterion = nn.CrossEntropyLoss()
        
        # Historia de entrenamiento
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        print(f"🖥️  Dispositivo: {device}")
        print(f"📊 Optimizer: Adam (lr={learning_rate}, wd={weight_decay})")
    
    def set_class_weights(self, class_counts: np.ndarray):
        """
        Establece pesos de clase para manejar desbalanceo.
        
        Args:
            class_counts: Array con conteo de muestras por clase
        """
        total = class_counts.sum()
        weights = total / (len(class_counts) * class_counts)
        weights = torch.FloatTensor(weights).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        print(f"⚖️  Class weights establecidos: {weights.cpu().numpy()}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Entrena una época.
        
        Args:
            dataloader: DataLoader de entrenamiento
            
        Returns:
            Tupla de (loss_promedio, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Actualizar progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Valida el modelo.
        
        Args:
            dataloader: DataLoader de validación
            
        Returns:
            Tupla de (loss_promedio, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(dataloader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 50,
             early_stopping_patience: int = 10,
             save_dir: str = None) -> Dict:
        """
        Entrena el modelo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número de épocas
            early_stopping_patience: Paciencia para early stopping
            save_dir: Directorio para guardar checkpoints
            
        Returns:
            Historia de entrenamiento
        """
        print(f"\n🏋️  Iniciando entrenamiento ({epochs} épocas)...\n")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Época {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validar
            val_loss, val_acc = self.validate(val_loader)
            
            # Actualizar learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Guardar historia
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Imprimir métricas
            print(f"\n📊 Resultados:")
            print(f"  • Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  • Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            print(f"  • Learning Rate: {current_lr:.6f}")
            
            # Calcular overfitting
            overfitting = train_acc - val_acc
            print(f"  • Overfitting: {overfitting*100:.2f}%")
            
            if overfitting > 0.05:
                print(f"  ⚠️  Advertencia: Overfitting > 5%")
            
            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
                
                if save_dir:
                    checkpoint_path = save_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"  ✅ Mejor modelo guardado (val_acc: {val_acc*100:.2f}%)")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹️  Early stopping activado (sin mejora en {early_stopping_patience} épocas)")
                break
        
        # Cargar mejor modelo
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ Cargado mejor modelo (val_acc: {self.best_val_acc*100:.2f}%)")
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total de entrenamiento: {training_time/60:.2f} minutos")
        
        return self.history
    
    def plot_training_history(self, save_path: str = None):
        """
        Plotea la historia de entrenamiento.
        
        Args:
            save_path: Ruta para guardar la figura
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Evolución del Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, [acc*100 for acc in self.history['train_acc']], 
                    'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, [acc*100 for acc in self.history['val_acc']], 
                    'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Evolución del Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figura guardada en: {save_path}")
        
        plt.close()  # Cerrar figura sin mostrar (no requiere display)
    
    def save_history(self, save_path: str):
        """Guarda la historia de entrenamiento."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✅ Historia guardada en: {save_path}")


if __name__ == '__main__':
    print("🧪 Este módulo debe importarse, no ejecutarse directamente")
