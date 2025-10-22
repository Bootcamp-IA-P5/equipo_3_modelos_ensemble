"""
Extractor de embeddings usando CNNs pre-entrenadas.
Optimizado para hardware con 6GB VRAM.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple
import numpy as np
from tqdm import tqdm


class CNNFeatureExtractor(nn.Module):
    """Extractor de características usando CNN pre-entrenada."""
    
    def __init__(self, 
                 model_name: str = 'resnet18',
                 pretrained: bool = True,
                 num_classes: int = 6):
        """
        Args:
            model_name: 'resnet18', 'mobilenet_v2', 'efficientnet_b0'
            pretrained: Usar pesos pre-entrenados de ImageNet
            num_classes: Número de clases de salida
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Cargar modelo pre-entrenado
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self.embedding_dim = 512
            # Modificar última capa
            self.backbone.fc = nn.Linear(self.embedding_dim, num_classes)
            
        elif model_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            self.embedding_dim = 1280
            # Modificar clasificador
            self.backbone.classifier[1] = nn.Linear(self.embedding_dim, num_classes)
            
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.embedding_dim = 1280
            # Modificar clasificador
            self.backbone.classifier[1] = nn.Linear(self.embedding_dim, num_classes)
        
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        print(f"✅ Modelo {model_name} cargado")
        print(f"   • Embedding dim: {self.embedding_dim}")
        print(f"   • Num classes: {num_classes}")
        print(f"   • Pretrained: {pretrained}")
    
    def forward(self, x):
        """Forward pass completo."""
        return self.backbone(x)
    
    def extract_embeddings(self, x):
        """
        Extrae embeddings de la penúltima capa.
        
        Args:
            x: Tensor de imágenes (B, C, H, W)
            
        Returns:
            Embeddings (B, embedding_dim)
        """
        if self.model_name == 'resnet18':
            # Para ResNet, extraer antes de la capa FC
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            embeddings = torch.flatten(x, 1)
            
        elif self.model_name in ['mobilenet_v2', 'efficientnet_b0']:
            # Para MobileNet/EfficientNet, extraer antes del clasificador
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            embeddings = torch.flatten(x, 1)
        
        return embeddings


class EmbeddingExtractor:
    """Utilidad para extraer embeddings de un dataset completo."""
    
    def __init__(self, 
                 model: CNNFeatureExtractor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Modelo CNN para extraer embeddings
            device: Dispositivo de cómputo
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        print(f"🖥️  Dispositivo: {device}")
    
    def extract_from_dataloader(self, dataloader):
        """
        Extrae embeddings de todo un dataloader.
        
        Args:
            dataloader: DataLoader de PyTorch
            
        Returns:
            Tupla de (embeddings, labels) como numpy arrays
        """
        all_embeddings = []
        all_labels = []
        
        print(f"\n🔍 Extrayendo embeddings...")
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Procesando batches"):
                images = images.to(self.device)
                
                # Extraer embeddings
                embeddings = self.model.extract_embeddings(images)
                
                # Mover a CPU y convertir a numpy
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
        
        # Concatenar todos los batches
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
        print(f"✅ Embeddings extraídos: {all_embeddings.shape}")
        
        return all_embeddings, all_labels


class SimpleClassifier(nn.Module):
    """Clasificador simple para fine-tuning."""
    
    def __init__(self, 
                 backbone: str = 'resnet18',
                 num_classes: int = 6,
                 dropout: float = 0.5):
        """
        Args:
            backbone: Modelo base
            num_classes: Número de clases
            dropout: Tasa de dropout
        """
        super(SimpleClassifier, self).__init__()
        
        # Cargar feature extractor
        self.feature_extractor = CNNFeatureExtractor(
            model_name=backbone,
            pretrained=True,
            num_classes=num_classes
        )
        
        # Congelar primeras capas para transfer learning
        self.freeze_backbone(freeze=True)
        
        self.dropout = nn.Dropout(dropout)
    
    def freeze_backbone(self, freeze: bool = True):
        """Congela/descongela el backbone."""
        if self.feature_extractor.model_name == 'resnet18':
            # Congelar todas las capas excepto la última
            for param in self.feature_extractor.backbone.parameters():
                param.requires_grad = not freeze
            
            # Descongelar capa de clasificación
            for param in self.feature_extractor.backbone.fc.parameters():
                param.requires_grad = True
        
        elif self.feature_extractor.model_name in ['mobilenet_v2', 'efficientnet_b0']:
            # Congelar features
            for param in self.feature_extractor.backbone.features.parameters():
                param.requires_grad = not freeze
            
            # Descongelar clasificador
            for param in self.feature_extractor.backbone.classifier.parameters():
                param.requires_grad = True
        
        status = "congelado" if freeze else "descongelado"
        print(f"🔒 Backbone {status}")
    
    def forward(self, x):
        """Forward pass."""
        return self.feature_extractor(x)
    
    def extract_embeddings(self, x):
        """Extrae embeddings."""
        return self.feature_extractor.extract_embeddings(x)


def get_model(model_name: str = 'resnet18', 
              num_classes: int = 6,
              pretrained: bool = True) -> SimpleClassifier:
    """
    Factory function para crear modelos.
    
    Args:
        model_name: Nombre del modelo
        num_classes: Número de clases
        pretrained: Usar pesos pre-entrenados
        
    Returns:
        Modelo instanciado
    """
    model = SimpleClassifier(
        backbone=model_name,
        num_classes=num_classes,
        dropout=0.5
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parámetros del modelo:")
    print(f"  • Total: {total_params:,}")
    print(f"  • Entrenables: {trainable_params:,}")
    print(f"  • Congelados: {total_params - trainable_params:,}")
    
    return model


if __name__ == '__main__':
    # Test del modelo
    print("🧪 Probando extractor de embeddings...")
    
    model = get_model('resnet18', num_classes=6)
    
    # Test con tensor aleatorio
    x = torch.randn(4, 3, 128, 128)
    
    # Forward pass
    output = model(x)
    print(f"\n✅ Output shape: {output.shape}")
    
    # Extraer embeddings
    embeddings = model.extract_embeddings(x)
    print(f"✅ Embeddings shape: {embeddings.shape}")
