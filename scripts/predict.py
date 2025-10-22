"""
Script de predicción para clasificación de residuos.
Se ejecuta desde la línea de comandos y devuelve resultados en JSON.
"""

import sys
import os

# Forzar UTF-8 para evitar errores con emojis en Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import argparse
import joblib

from src.models.cnn_extractor import get_model


class SingleImageDataset(Dataset):
    """Dataset para una sola imagen."""
    
    def __init__(self, image_path: str, transform=None):
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Label dummy


class GarbageClassifier:
    """Clasificador de residuos."""
    
    CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    RECYCLING_INFO = {
        'cardboard': {
            'name': 'Cartón',
            'bin': '📦 Contenedor Azul',
            'tips': 'Asegúrate de que esté limpio y plegado.'
        },
        'glass': {
            'name': 'Vidrio',
            'bin': '🍾 Contenedor Verde',
            'tips': 'Enjuaga antes de reciclar. No incluir espejos o cristal.'
        },
        'metal': {
            'name': 'Metal',
            'bin': '🥫 Contenedor Amarillo',
            'tips': 'Latas y envases metálicos. Aplastar para ahorrar espacio.'
        },
        'paper': {
            'name': 'Papel',
            'bin': '📄 Contenedor Azul',
            'tips': 'Solo papel limpio y seco. No papel sucio o encerado.'
        },
        'plastic': {
            'name': 'Plástico',
            'bin': '🥤 Contenedor Amarillo',
            'tips': 'Botellas y envases plásticos. Vaciar y enjuagar.'
        },
        'trash': {
            'name': 'Basura General',
            'bin': '🗑️ Contenedor Gris/Negro',
            'tips': 'Residuos no reciclables. Considerar reducir este tipo de residuos.'
        }
    }
    
    def __init__(self, 
                 cnn_model_path: str,
                 ensemble_model_path: str,
                 model_type: str = 'resnet18',
                 img_size: int = 128,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            cnn_model_path: Ruta al modelo CNN guardado
            ensemble_model_path: Ruta al modelo ensemble guardado
            model_type: Tipo de modelo CNN
            img_size: Tamaño de imagen
            device: Dispositivo de cómputo
        """
        self.device = device
        self.img_size = img_size
        
        # Cargar modelo CNN (silenciar prints enviándolos a stderr)
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            self.cnn_model = get_model(model_type, num_classes=6, pretrained=False)
        
        if Path(cnn_model_path).exists():
            checkpoint = torch.load(cnn_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Intentar cargar el state_dict directamente
                try:
                    self.cnn_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                except Exception as e:
                    print(f"[WARNING] Error al cargar modelo: {e}", file=sys.stderr)
                    # Si falla, intentar sin strict matching
                    state_dict = checkpoint['model_state_dict']
                    # Filtrar solo las claves que coinciden
                    model_dict = self.cnn_model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.cnn_model.load_state_dict(model_dict)
            else:
                self.cnn_model.load_state_dict(checkpoint, strict=False)
        
        self.cnn_model.to(device)
        self.cnn_model.eval()
        
        # Cargar modelo ensemble
        self.ensemble_model = joblib.load(ensemble_model_path)
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Imprimir a stderr para no interferir con JSON en stdout
        print(f"[OK] Modelos cargados (device: {device})", file=sys.stderr)
    
    def predict(self, image_path: str) -> dict:
        """
        Predice la clase de una imagen.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Diccionario con predicción e información
        """
        # Cargar imagen
        dataset = SingleImageDataset(image_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=1)
        
        # Extraer embedding
        with torch.no_grad():
            for image, _ in loader:
                image = image.to(self.device)
                embedding = self.cnn_model.extract_embeddings(image)
                embedding = embedding.cpu().numpy()
        
        # Predecir con ensemble
        prediction_idx = self.ensemble_model.predict(embedding)[0]
        probabilities = self.ensemble_model.predict_proba(embedding)[0]
        
        predicted_class = self.CLASSES[prediction_idx]
        confidence = probabilities[prediction_idx]
        
        # Información de reciclaje
        recycling_info = self.RECYCLING_INFO[predicted_class]
        
        return {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'probabilities': {cls: float(prob) for cls, prob in zip(self.CLASSES, probabilities)},
            'recycling_info': recycling_info
        }


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Predecir clase de residuo')
    parser.add_argument('--image', type=str, required=True,
                       help='Ruta a la imagen')
    parser.add_argument('--cnn_model', type=str, 
                       default='models/trained/best_model.pth',
                       help='Ruta al modelo CNN')
    parser.add_argument('--ensemble_model', type=str,
                       default='models/trained/stacking_model.pkl',
                       help='Ruta al modelo ensemble')
    parser.add_argument('--model_type', type=str, default='resnet18',
                       help='Tipo de modelo CNN')
    
    args = parser.parse_args()
    
    try:
        # Crear clasificador
        classifier = GarbageClassifier(
            cnn_model_path=args.cnn_model,
            ensemble_model_path=args.ensemble_model,
            model_type=args.model_type
        )
        
        # Predecir
        result = classifier.predict(args.image)
        
        # Imprimir resultado como JSON (ensure_ascii=False para Unicode)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        # En caso de error, devolver error en formato JSON
        error_result = {
            'error': str(e),
            'prediction': 'unknown',
            'confidence': 0.0
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == '__main__':
    main()
