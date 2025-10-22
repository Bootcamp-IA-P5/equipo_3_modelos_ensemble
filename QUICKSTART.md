# Guía de Inicio Rápido - EcoSort

## 🚀 Instalación y Ejecución

### Opción 1: Instalación Local

#### 1. Clonar repositorio
```bash
git clone https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble.git
cd equipo_3_modelos_ensemble
```

#### 2. Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

#### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

#### 4. Descargar dataset
Coloca el dataset en `data/raw/garbage_classification/`

O descárgalo automáticamente:
```bash
# Configura tu API key de Kaggle primero
python scripts/download_dataset.py
```

#### 5. Entrenar modelos
```bash
# Entrenamiento completo (CNN + Ensemble)
python scripts/train_models.py --epochs 30 --batch_size 32

# Con optimización de hiperparámetros
python scripts/train_models.py --epochs 30 --optimize_hp --hp_trials 30
```

#### 6. Lanzar aplicación
```bash
streamlit run app/app_streamlit.py
```

Abre tu navegador en: http://localhost:8501

---

### Opción 2: Docker

#### 1. Construir imagen
```bash
docker build -t ecosort:latest .
```

#### 2. Ejecutar contenedor
```bash
docker run -p 8501:8501 \
  -v $(pwd)/models/trained:/app/models/trained:ro \
  -v $(pwd)/data/feedback:/app/data/feedback \
  ecosort:latest
```

#### 3. O usar Docker Compose
```bash
docker-compose up -d
```

Abre tu navegador en: http://localhost:8501

---

## 📊 Uso del Sistema

### Clasificar Imágenes

1. Abre la aplicación web
2. Sube una imagen de un residuo
3. Haz clic en "Clasificar Residuo"
4. Revisa los resultados:
   - Predicción
   - Confianza
   - Contenedor de reciclaje correcto
   - Consejos de reciclaje

### Proporcionar Feedback

1. Después de clasificar, indica si la predicción es correcta
2. Si es incorrecta, selecciona la clase correcta
3. Opcionalmente, añade comentarios
4. El feedback se guarda automáticamente para reentrenamiento

### Ver Métricas

1. Ve a la página "Métricas" en el menú lateral
2. Revisa:
   - Accuracy en tiempo real
   - Distribución de predicciones
   - Evolución temporal
   - Feedback recibido

---

## 🧪 Testing

### Ejecutar tests
```bash
pytest tests/ -v
```

### Con cobertura
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📝 Ejemplos de Uso

### Predicción desde línea de comandos
```bash
python scripts/predict.py \
  --image data/test/sample.jpg \
  --cnn_model models/trained/best_model.pth \
  --ensemble_model models/trained/stacking_model.pkl
```

### Exportar feedback para reentrenamiento
```python
from pathlib import Path
import json
import pandas as pd

# Cargar feedback
feedback_files = list(Path('data/feedback').glob('feedback_*.json'))
feedbacks = [json.load(open(f)) for f in feedback_files]

# Crear DataFrame
df = pd.DataFrame(feedbacks)

# Filtrar incorrectos para reentrenamiento
incorrect = df[df['correct'] == False]
incorrect.to_csv('data/feedback/for_retraining.csv', index=False)
```

---

## 🔧 Configuración Avanzada

### Ajustar tamaño de imagen (para más/menos VRAM)

En `scripts/train_models.py`:
```bash
# Para 4GB VRAM
python scripts/train_models.py --img_size 96 --batch_size 24

# Para 8GB+ VRAM
python scripts/train_models.py --img_size 224 --batch_size 64
```

### Usar diferentes modelos CNN

```bash
python scripts/train_models.py --model_name mobilenet_v2
# Opciones: resnet18, mobilenet_v2, efficientnet_b0
```

---

## ❓ Solución de Problemas

### Error: CUDA out of memory
```bash
# Reducir batch size o tamaño de imagen
python scripts/train_models.py --img_size 96 --batch_size 16
```

### Error: ModuleNotFoundError
```bash
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Aplicación no carga modelos
```bash
# Verificar que existan los modelos
ls models/trained/

# Si no existen, entrenar primero
python scripts/train_models.py
```

---

## 📞 Soporte

¿Problemas o preguntas?

- 📧 Email: equipo3@bootcamp-f5.com
- 🐛 Issues: [GitHub Issues](https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble/issues)
- 📚 Docs: Ver `README.md` completo

---

**Equipo 3 - Bootcamp F5 IA** | ♻️ EcoSort
