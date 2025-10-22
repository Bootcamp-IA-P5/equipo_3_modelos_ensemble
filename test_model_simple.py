"""
Script simple para probar el modelo de clasificación sin interfaz
"""
import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from models.cnn_extractor import CNNFeatureExtractor
from data.preprocessing import DataPreprocessor
import pickle

def test_model():
    print("=" * 60)
    print("TEST SIMPLE DEL MODELO")
    print("=" * 60)
    
    # Rutas
    project_root = Path(__file__).parent
    
    # 1. Buscar el experimento más reciente
    experiments_dir = project_root / "models" / "trained"
    if not experiments_dir.exists():
        print("❌ No existe el directorio de modelos entrenados")
        return
    
    experiments = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
    if not experiments:
        print("❌ No hay experimentos entrenados")
        return
    
    latest_exp = experiments[-1]
    print(f"✅ Experimento encontrado: {latest_exp.name}")
    
    # 2. Cargar modelo CNN
    cnn_model_path = latest_exp / "best_model.pth"
    if not cnn_model_path.exists():
        print(f"❌ No existe {cnn_model_path}")
        return
    
    print(f"\n📦 Cargando modelo CNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    cnn_extractor = CNNFeatureExtractor(model_name="resnet18", num_classes=6)
    checkpoint = torch.load(cnn_model_path, map_location=device)
    
    # Remover el prefijo 'feature_extractor.' de las claves
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('feature_extractor.'):
            new_key = key.replace('feature_extractor.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    cnn_extractor.load_state_dict(new_state_dict)
    cnn_extractor.to(device)
    cnn_extractor.eval()
    print("   ✅ Modelo CNN cargado")
    
    # 3. Cargar modelo ensemble (probamos con stacking)
    ensemble_model_path = latest_exp / "stacking_model.pkl"
    if not ensemble_model_path.exists():
        print(f"❌ No existe {ensemble_model_path}")
        return
    
    print(f"\n📦 Cargando modelo ensemble...")
    with open(ensemble_model_path, 'rb') as f:
        ensemble_model = pickle.load(f)
    print("   ✅ Modelo ensemble cargado")
    
    # 4. Cargar una imagen de prueba del dataset
    test_images_dir = project_root / "data" / "raw" / "garbage_classification" / "cardboard"
    if not test_images_dir.exists():
        print(f"❌ No existe {test_images_dir}")
        return
    
    test_images = list(test_images_dir.glob("*.jpg"))
    if not test_images:
        print("❌ No hay imágenes de prueba")
        return
    
    test_image_path = test_images[0]
    print(f"\n🖼️  Imagen de prueba: {test_image_path.name}")
    
    # 5. Preprocesar imagen
    print(f"\n⚙️  Preprocesando imagen...")
    preprocessor = DataPreprocessor(target_size=(128, 128))
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = preprocessor.preprocess_image(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    print(f"   Shape: {image_tensor.shape}")
    
    # 6. Extraer embeddings
    print(f"\n🔍 Extrayendo embeddings...")
    with torch.no_grad():
        embeddings = cnn_extractor.extract_embeddings(image_tensor)
        embeddings_np = embeddings.cpu().numpy()
    print(f"   Embeddings shape: {embeddings_np.shape}")
    
    # 7. Predecir con ensemble
    print(f"\n🎯 Prediciendo clase...")
    prediction_idx = ensemble_model.predict(embeddings_np)[0]
    
    # Obtener probabilidades si el modelo las soporta
    if hasattr(ensemble_model, 'predict_proba'):
        probabilities = ensemble_model.predict_proba(embeddings_np)[0]
    else:
        probabilities = None
    
    # 8. Mapear a clase
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    predicted_class = class_names[prediction_idx]
    
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"📍 Imagen real: cardboard")
    print(f"🎯 Predicción: {predicted_class}")
    if probabilities is not None:
        print(f"\n📊 Probabilidades:")
        for i, class_name in enumerate(class_names):
            print(f"   {class_name:12s}: {probabilities[i]*100:5.2f}%")
    print("=" * 60)
    
    if predicted_class == "cardboard":
        print("✅ ¡PREDICCIÓN CORRECTA!")
    else:
        print("❌ Predicción incorrecta")

if __name__ == "__main__":
    test_model()
