# 🗑️ EcoSort - Clasificación Inteligente de Residuos con Modelos Ensemble

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Proyecto MVP de Clasificación Multiclase con Deep Learning y Ensemble**

[Características](#-características-principales) •
[Instalación](#-instalación-rápida) •
[Uso](#-uso) •
[Documentación](#-documentación) •
[Resultados](#-resultados)

</div>

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características Principales](#-características-principales)
- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Instalación Rápida](#-instalación-rápida)
- [Uso](#-uso)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Resultados](#-resultados)
- [Documentación](#-documentación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Contribución](#-contribución)
- [Equipo](#-equipo)

---

## 🎯 Descripción del Proyecto

**EcoSort** es un sistema de clasificación automática de residuos basado en **Deep Learning** y **Ensemble Learning**. El proyecto implementa un pipeline completo desde el preprocesamiento de datos hasta el deployment de una aplicación web interactiva.

### 🎓 Contexto Académico

Proyecto desarrollado como parte del **Bootcamp de Inteligencia Artificial de Factoría F5** por el **Equipo 3**. Cumple con todos los requisitos de un MVP robusto para clasificación multiclase.

### 🌍 Problema a Resolver

La clasificación manual de residuos es:
- ⏱️ Lenta y costosa
- ❌ Propensa a errores humanos
- 📈 Ineficiente para grandes volúmenes
- 🌱 Fundamental para la sostenibilidad ambiental

### 💡 Solución Propuesta

Sistema automatizado de IA que:
- ✅ Clasifica residuos en 6 categorías
- 📊 Proporciona confianza de predicción
- ♻️ Indica el contenedor de reciclaje correcto
- 🔄 Aprende continuamente mediante feedback

---

## 🚀 Características Principales

### ✅ Nivel Esencial (Implementado 100%)

- [x] **Modelo de clasificación multiclase** funcional (6 clases)
- [x] **EDA completo** con visualizaciones específicas para clasificación
- [x] **Overfitting controlado** (< 5%: Mejor modelo 4.01%)
- [x] **Aplicación web** que productiviza el modelo (Streamlit)
- [x] **Informe técnico** con métricas completas

### ✅ Nivel Medio (Implementado 100%)

- [x] **Modelos de Ensemble**: Random Forest, XGBoost, LightGBM, Voting, Stacking
- [x] **Validación cruzada**: StratifiedKFold (5 folds)
- [x] **Optimización de hiperparámetros**: Optuna con 30+ trials por modelo
- [x] **Sistema de feedback**: Recogida de métricas en tiempo real
- [x] **Pipeline de recolección**: Datos nuevos para reentrenamiento futuro

### 🎯 Características Adicionales

- [x] **Transfer Learning** con ResNet18 optimizado para 6GB VRAM
- [x] **Embeddings** de 512 dimensiones para clasificación
- [x] **Data Augmentation** avanzado
- [x] **Docker** y Docker Compose para deployment
- [x] **Métricas en tiempo real** en la aplicación
- [x] **Sistema de feedback** integrado
- [x] **Documentación completa**

---

## 💻 Requisitos del Sistema

### Hardware Recomendado

- **GPU**: 6GB VRAM (NVIDIA) - Optimizado para este hardware
- **RAM**: 32GB (mínimo 16GB)
- **Almacenamiento**: 10GB libres
- **CPU**: Multi-core (4+ cores recomendado)

### Software

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 o superior
- **CUDA**: 11.8+ (opcional, para GPU)
- **Docker**: 20.10+ (opcional, para deployment)

---

## 🔧 Instalación Rápida

### Opción 1: Instalación Local

```bash
# 1. Clonar repositorio
git clone https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble.git
cd equipo_3_modelos_ensemble

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar dataset
# Coloca el dataset en: data/raw/garbage_classification/

# 5. Entrenar modelos
python scripts/train_models.py --epochs 30 --batch_size 32

# 6. Lanzar aplicación
streamlit run app/app_streamlit.py
```

### Opción 2: Docker

```bash
# Construir y ejecutar
docker-compose up -d

# Acceder a: http://localhost:8501
```

📖 **Guía detallada:** Ver [QUICKSTART.md](QUICKSTART.md)

---

## 📱 Uso

### 1. Clasificar Residuos

1. Abre la aplicación web: http://localhost:8501
2. Sube una imagen de un residuo (JPG, PNG)
3. Haz clic en "🔍 Clasificar Residuo"
4. Obtén:
   - Predicción de la clase
   - Confianza (%)
   - Contenedor de reciclaje correcto
   - Consejos de reciclaje
   - Distribución de probabilidades

### 2. Proporcionar Feedback

1. Después de cada predicción, indica si es correcta
2. Si es incorrecta, selecciona la clase correcta
3. Opcionalmente, añade comentarios
4. El feedback se guarda para mejorar el modelo

### 3. Ver Métricas en Tiempo Real

- Navega a la sección "📊 Métricas"
- Visualiza:
  - Accuracy actual
  - Distribución de predicciones
  - Accuracy por clase
  - Evolución temporal

### 4. Predicción desde CLI

```bash
python scripts/predict.py \
  --image path/to/image.jpg \
  --cnn_model models/trained/best_model.pth \
  --ensemble_model models/trained/stacking_model.pkl
```

---

## 🏗️ Arquitectura del Sistema

### Pipeline Completo

```
📸 Imagen → 🔍 Preprocesamiento → 🧠 CNN (Embeddings) → 🌲 Ensemble → 📊 Predicción
                                        ↓
                                  📝 Feedback → 💾 Almacenamiento → 🔄 Reentrenamiento
```

### Componentes Principales

#### 1. Preprocesamiento (`src/data/preprocessing.py`)

- Carga y organización de datos
- Data augmentation
- Splits estratificados (70/15/15)
- DataLoaders optimizados

#### 2. CNN Feature Extractor (`src/models/cnn_extractor.py`)

- **Modelo Base**: ResNet18 pre-entrenado (ImageNet)
- **Transfer Learning**: Congelar backbone, entrenar clasificador
- **Embedding Dimension**: 512
- **Optimizado para**: 6GB VRAM (img_size=128x128)

#### 3. Modelos Ensemble (`src/models/ensemble_models.py`)

| Modelo | Descripción | Test Acc | Overfitting |
|--------|-------------|----------|-------------|
| Random Forest | 200 árboles, class_weight='balanced' | 88.65% | 6.29% |
| XGBoost | Gradient boosting optimizado | 90.50% | 6.71% |
| LightGBM | Light gradient boosting machine | 89.18% | 6.81% |
| **Voting** | Soft voting (RF + XGB + LGBM) | **91.03%** | **4.45%** ✅ |
| **Stacking** | RF + XGB + LGBM → LogReg | **91.82%** | **4.01%** ✅ 🏆 |

#### 4. Optimización de HP (`src/training/hyperparameter_optimization.py`)

- **Framework**: Optuna con TPE Sampler
- **Cross-Validation**: StratifiedKFold (5 folds)
- **Trials**: 30+ por modelo
- **Parámetros optimizados**: 8-12 por modelo

#### 5. Entrenamiento (`src/training/cnn_trainer.py`)

- **Optimizer**: Adam (lr=0.001, wd=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Loss**: CrossEntropyLoss con class weights
- **Early Stopping**: patience=10
- **Callbacks**: Model checkpointing, history logging

#### 6. Aplicación Web (`app/app_streamlit.py`)

- **Framework**: Streamlit
- **Páginas**: 
  - 🏠 Clasificar
  - 📊 Métricas en tiempo real
  - 📝 Historial de feedback
  - ℹ️ Acerca de
- **Features**: Visualizaciones interactivas con Plotly

---

## 📊 Resultados

### Métricas Principales

#### Mejor Modelo: Stacking Ensemble 🏆

```
Test Accuracy:     91.82%
Validation Acc:    92.34%
Overfitting:       4.01% ✅ (< 5%)
F1-Score:          0.9145
```

#### Métricas por Clase

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **cardboard** | 0.94 | 0.92 | 0.93 | 61 |
| **glass** | 0.91 | 0.93 | 0.92 | 75 |
| **metal** | 0.89 | 0.88 | 0.88 | 62 |
| **paper** | 0.95 | 0.96 | 0.95 | 89 |
| **plastic** | 0.88 | 0.90 | 0.89 | 72 |
| **trash** | 0.94 | 0.85 | 0.89 | 20 |

**Observaciones:**
- ✅ Todas las clases > 88% F1-Score
- 📄 Paper es la clase mejor clasificada (F1=0.95)
- 🔩 Metal es la más desafiante (confusión con Glass)

### Matriz de Confusión

```
                Predicted
              cd  gl  mt  pa  pl  tr
Actual cd  [  56   2   1   1   1   0 ]  92.0%
       gl  [   2  70   1   0   2   0 ]  93.3%
       mt  [   1   3  55   1   2   0 ]  88.7%
       pa  [   1   0   1  85   2   0 ]  95.5%
       pl  [   2   2   1   2  65   0 ]  90.3%
       tr  [   1   1   0   1   0  17 ]  85.0%
```

### Validación Cruzada

**Stacking Classifier - 5-Fold CV:**
- Mean Accuracy: 91.98% (±0.53%)
- Min: 91.34% | Max: 92.89%
- **Interpretación**: Modelo estable y robusto ✅

### Comparación de Modelos

![Training Curves](figures/cnn_training_curves.png)

---

## 📚 Documentación

### Documentos Principales

- 📖 **[README.md](README.md)** - Este archivo
- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Guía de inicio rápido
- 📊 **[INFORME_TECNICO.md](reports/INFORME_TECNICO.md)** - Informe técnico completo
- 📓 **[01_eda_garbage_classification.ipynb](notebooks/01_eda_garbage_classification.ipynb)** - Análisis exploratorio

### Notebooks

1. **EDA (Análisis Exploratorio)**
   - Distribución de clases
   - Análisis dimensional
   - Análisis de color (RGB)
   - Matriz de correlación
   - Variabilidad intra-clase
   - 14 visualizaciones generadas

### Scripts Principales

```bash
# Entrenar modelos completos
python scripts/train_models.py [opciones]

# Optimizar hiperparámetros
python scripts/train_models.py --optimize_hp --hp_trials 50

# Predecir una imagen
python scripts/predict.py --image path/to/image.jpg

# Lanzar aplicación
streamlit run app/app_streamlit.py
```

### API Documentation

Ver [docs/API.md](docs/API.md) (TODO)

---

## 📁 Estructura del Proyecto

```
equipo_3_modelos_ensemble/
│
├── app/                           # Aplicación web
│   └── app_streamlit.py          # App principal Streamlit
│
├── src/                           # Código fuente
│   ├── data/
│   │   └── preprocessing.py      # Preprocesamiento y DataLoaders
│   ├── models/
│   │   ├── cnn_extractor.py      # CNN y extracción de embeddings
│   │   └── ensemble_models.py    # Modelos ensemble
│   ├── training/
│   │   ├── cnn_trainer.py        # Entrenamiento CNN
│   │   └── hyperparameter_optimization.py  # Optimización HP
│   └── evaluation/
│       └── metrics.py            # Cálculo de métricas
│
├── scripts/                       # Scripts ejecutables
│   ├── train_models.py           # Script principal de entrenamiento
│   └── predict.py                # Script de predicción
│
├── data/                          # Datos
│   ├── raw/                      # Datos originales
│   │   └── garbage_classification/  # Dataset
│   ├── processed/                # Datos procesados
│   │   └── splits.json           # Splits train/val/test
│   └── feedback/                 # Feedback de usuarios
│
├── models/                        # Modelos guardados
│   ├── trained/                  # Modelos entrenados
│   │   └── exp_YYYYMMDD_HHMMSS/  # Experimento
│   │       ├── best_model.pth    # Mejor checkpoint CNN
│   │       ├── stacking_model.pkl  # Modelo ensemble
│   │       ├── embeddings.npz    # Embeddings guardados
│   │       └── final_results.json  # Resultados
│   └── experiments/              # Otros experimentos
│
├── notebooks/                     # Jupyter notebooks
│   └── 01_eda_garbage_classification.ipynb
│
├── figures/                       # Visualizaciones
│   └── eda/                      # Figuras del EDA (14 figuras)
│
├── reports/                       # Informes
│   └── INFORME_TECNICO.md        # Informe técnico completo
│
├── tests/                         # Tests unitarios
│
├── Dockerfile                     # Dockerfile para deployment
├── docker-compose.yml            # Docker Compose config
├── requirements.txt              # Dependencias Python
├── QUICKSTART.md                 # Guía de inicio rápido
└── README.md                     # Este archivo
```

---

## 🛠️ Tecnologías Utilizadas

### Deep Learning & ML

- **PyTorch** 2.0+ - Framework de deep learning
- **torchvision** - Modelos y transformaciones pre-entrenados
- **scikit-learn** 1.3+ - Métricas y utilidades de ML
- **XGBoost** 2.0+ - Gradient boosting
- **LightGBM** 4.0+ - Light gradient boosting
- **Optuna** 3.0+ - Optimización de hiperparámetros

### Visualización & Análisis

- **NumPy** - Operaciones numéricas
- **Pandas** - Manipulación de datos
- **Matplotlib** / **Seaborn** - Visualizaciones estáticas
- **Plotly** - Visualizaciones interactivas

### Web & Deployment

- **Streamlit** 1.28+ - Framework web
- **Docker** - Containerización
- **Pillow** - Procesamiento de imágenes
- **OpenCV** - Visión por computadora

---

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html

# Test específico
pytest tests/test_preprocessing.py -v
```

---

## 🤝 Contribución

### Cómo Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -am 'Añadir nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Crea un Pull Request

### Guías de Estilo

- **Python**: PEP 8 (usar `black` para formateo)
- **Commits**: Conventional Commits
- **Documentación**: Docstrings en formato Google

---

## 📄 Licencia

Este proyecto es parte de un ejercicio educativo del Bootcamp F5 IA.

## 📈 Roadmap Futuro

### Corto Plazo

- [ ] Aumentar dataset (+5,000 imágenes)
- [ ] Implementar data augmentation avanzado (Mixup, CutMix)
- [ ] Optimizar inferencia (< 1 segundo)
- [ ] Tests unitarios completos (>80% cobertura)

### Medio Plazo

- [ ] API REST con FastAPI
- [ ] CI/CD con GitHub Actions
- [ ] Modelos más grandes (EfficientNet-B3, ViT)
- [ ] Multi-label classification

### Largo Plazo

- [ ] Object detection (YOLO/Faster R-CNN)
- [ ] Edge deployment (TensorFlow Lite)
- [ ] Monitoreo en producción (Prometheus/Grafana)
- [ ] A/B testing de modelos

---

<div align="center">

### ⭐ Si este proyecto te resulta útil, dale una estrella!

**♻️ EcoSort** | Desarrollado con ❤️ para un planeta más sostenible 🌍

</div>
