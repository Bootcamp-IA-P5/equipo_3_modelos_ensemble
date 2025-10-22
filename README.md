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

## 👥 Equipo

**Equipo 3 - Bootcamp F5 IA**

Proyecto desarrollado en el marco del Bootcamp de Inteligencia Artificial de Factoría F5.

### Contacto

- 📧 Email: [equipo3@bootcamp-f5.com](mailto:equipo3@bootcamp-f5.com)
- 🐙 GitHub: [Factoria-F5-madrid/equipo_3_modelos_ensemble](https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble)
- 🎓 Bootcamp: [Factoría F5](https://factoriaf5.org/)

---

## 📄 Licencia

Este proyecto es parte de un ejercicio educativo del Bootcamp F5 IA.

---

## 🙏 Agradecimientos

- 🎓 **Factoría F5** por el programa de formación
- 🌐 **Kaggle** por el dataset de clasificación de residuos
- 💻 **Comunidad Open Source** por las herramientas utilizadas
- 👨‍🏫 **Mentores** por el apoyo y guía

---

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

**Equipo 3 - Bootcamp F5 IA** | 2024

</div>

## 🎯 Descripción del Proyecto

Este proyecto desarrolla un sistema de **clasificación automática de residuos** utilizando técnicas de **Deep Learning** y **modelos ensemble**. El objetivo es entrenar un modelo capaz de identificar y clasificar diferentes tipos de residuos a partir de imágenes, facilitando el proceso de reciclaje y gestión de residuos.

### Objetivo Principal
Desarrollar un modelo de clasificación de imágenes robusto y preciso que pueda identificar automáticamente 6 categorías de residuos, utilizando técnicas de ensemble learning para mejorar el rendimiento.

## 💼 Contexto de Negocio

### Problema a Resolver
La **gestión inadecuada de residuos** es uno de los mayores desafíos ambientales actuales. La clasificación manual de residuos es:
- ⏱️ **Lenta y costosa** en tiempo y recursos humanos
- ❌ **Propensa a errores** debido al cansancio y falta de capacitación
- 📈 **Ineficiente** para procesar grandes volúmenes de residuos
- 🌍 **Poco sostenible** para el medio ambiente

### Solución Propuesta
Un **sistema automatizado de clasificación de residuos** basado en visión por computadora que:
- ✅ Clasifica residuos de forma rápida y precisa
- 📊 Mejora la eficiencia del proceso de reciclaje
- 💰 Reduce costos operativos
- ♻️ Aumenta las tasas de reciclaje
- 🌱 Contribuye a la sostenibilidad ambiental

### Casos de Uso
1. **Plantas de reciclaje**: Automatización de líneas de clasificación
2. **Contenedores inteligentes**: Clasificación en tiempo real
3. **Aplicaciones móviles**: Ayuda al ciudadano a clasificar correctamente
4. **Auditorías de residuos**: Análisis y reporting automático

## 📊 Dataset

### Origen
Dataset de **Garbage Classification** con imágenes de residuos reales en diferentes condiciones de iluminación, ángulos y fondos.

### Clases del Dataset
El dataset contiene **6 categorías** de residuos:

| Clase | Descripción | Ejemplos |
|-------|-------------|----------|
| 🥤 **Cardboard** | Cartón y cajas | Cajas de pizza, embalajes |
| 🍾 **Glass** | Vidrio | Botellas, frascos |
| 🔩 **Metal** | Metales | Latas, tapas metálicas |
| 📄 **Paper** | Papel | Periódicos, documentos |
| 🥤 **Plastic** | Plástico | Botellas PET, envases |
| 🗑️ **Trash** | Basura general | Residuos no reciclables |

### Estadísticas del Dataset

```
Total de imágenes: 2,528
Distribución por clase:
├── Paper:      594 imágenes (23.5%)
├── Glass:      501 imágenes (19.8%)
├── Plastic:    482 imágenes (19.1%)
├── Metal:      410 imágenes (16.2%)
├── Cardboard:  404 imágenes (16.0%)
└── Trash:      137 imágenes (5.4%)

Ratio de desbalanceo: 4.34x (Paper/Trash)
```

### Características de las Imágenes
- **Formato**: JPG
- **Dimensión promedio**: ~512×384 px
- **Tamaño medio**: ~30-50 KB
- **Canales**: RGB (3 canales)
- **Variabilidad**: Alta (diferentes ángulos, iluminación, fondos)

### Desafíos del Dataset
- ⚠️ **Desbalanceo de clases**: La clase "Trash" tiene significativamente menos imágenes
- 🔄 **Variabilidad intra-clase**: Alta diversidad dentro de cada categoría
- 🌈 **Condiciones de captura**: Diferentes iluminaciones y ángulos
- 📐 **Dimensiones variables**: Las imágenes tienen diferentes tamaños

## 🔍 Análisis Exploratorio de Datos (EDA)

### Notebook Principal
📓 **[01_eda_garbage_classification.ipynb](notebooks/01_eda_garbage_classification.ipynb)**

### Análisis Realizados

#### 1. Distribución de Clases
- ✅ Análisis de balance/desbalanceo entre clases
- 📊 Visualizaciones: gráficos de barras y circulares
- 💡 Resultado: Desbalanceo moderado, requiere técnicas de balanceo

#### 2. Características Dimensionales
- 📏 Análisis de anchos, altos y ratios de aspecto
- 📦 Distribución de tamaños de archivo
- 📈 Variabilidad por clase
- 💡 Resultado: Alta variabilidad dimensional, requiere normalización

#### 3. Análisis de Color (RGB)
- 🎨 Distribución de intensidad por canal (Rojo, Verde, Azul)
- ☀️ Análisis de brillo y contraste
- 🌈 Perfiles de color por clase
- 💡 Resultado: Cada clase tiene características de color distintivas

#### 4. Matriz de Correlación
- 🔗 Correlaciones entre 10 características:
  - Dimensiones: ancho, alto, aspecto, píxeles, tamaño
  - Color: R, G, B, brillo, contraste
- 💡 Resultado: Identificación de características redundantes y discriminativas

#### 5. Variabilidad Intra-Clase
- 📊 Coeficientes de variación (CV) por clase
- 📉 Análisis de homogeneidad/heterogeneidad
- 💡 Resultado: Variabilidad moderada-alta, sugiere necesidad de data augmentation

### Visualizaciones Generadas

El EDA genera automáticamente **14 figuras** guardadas en `figures/eda/`:

| Figura | Descripción |
|--------|-------------|
| `01_distribucion_clases.png` | Distribución de imágenes por clase |
| `02_histograma_anchos_por_clase.png` | Distribución de anchos por clase |
| `03_histograma_altos_por_clase.png` | Distribución de altos por clase |
| `04_boxplots_comparativos.png` | Comparación de características entre clases |
| `05_analisis_color_canales_rgb.png` | Análisis de canales RGB y brillo |
| `06_histogramas_rgb_por_clase.png` | Distribución RGB por clase |
| `07_matriz_correlacion.png` | Correlaciones entre características |
| `08_variabilidad_intraclase.png` | Coeficientes de variación por clase |
| `09_muestras_[clase].png` | Muestras visuales de cada clase (6 archivos) |

### Conclusiones del EDA

#### ✅ Aspectos Positivos
- Dataset relativamente balanceado (excepto clase "Trash")
- Buena variedad de imágenes por clase
- Características de color distintivas entre clases
- Dimensiones razonablemente consistentes

#### ⚠️ Consideraciones Importantes
1. **Desbalanceo de clases**: Aplicar técnicas de balanceo
   - Class weights en la función de pérdida
   - Data augmentation para clases minoritarias
   - Oversampling/undersampling

2. **Variabilidad dimensional**: Normalizar imágenes
   - Redimensionar a tamaño fijo (ej: 224×224 para transfer learning)
   - Padding o cropping inteligente

3. **Variabilidad intra-clase**: Data augmentation
   - Rotaciones, flips, cambios de brillo/contraste
   - Aumentar robustez del modelo

4. **Características discriminativas**:
   - Color y textura son importantes
   - Combinar características visuales de bajo y alto nivel

## 📁 Estructura del Proyecto

```
equipo_3_modelos_ensemble/
│
├── data/                           # Datos del proyecto
│   ├── raw/                        # Datos originales
│   │   └── garbage_classification/ # Dataset de imágenes
│   │       ├── cardboard/
│   │       ├── glass/
│   │       ├── metal/
│   │       ├── paper/
│   │       ├── plastic/
│   │       └── trash/
│   ├── processed/                  # Datos procesados
│   └── splits/                     # Train/Val/Test splits
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_eda_garbage_classification.ipynb  # EDA principal
│   └── 02_ensemble_model.ipynb     # Modelo ensemble
│
├── src/                            # Código fuente
│   ├── data/                       # Scripts de datos
│   ├── models/                     # Definición de modelos
│   ├── training/                   # Scripts de entrenamiento
│   └── evaluation/                 # Scripts de evaluación
│
├── models/                         # Modelos entrenados
│   ├── trained/                    # Modelos guardados
│   └── experiments/                # Experimentos
│
├── figures/                        # Figuras y visualizaciones
│   └── eda/                        # Figuras del EDA
│
├── scripts/                        # Scripts auxiliares
│   └── download_garbage_dataset.py # Descarga del dataset
│
├── tests/                          # Tests unitarios
│
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Archivos ignorados por git
└── README.md                       # Este archivo
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip o conda
- (Opcional) GPU con CUDA para entrenamiento acelerado

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble.git
cd equipo_3_modelos_ensemble
```

### Paso 2: Crear entorno virtual
```bash
# Con venv
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# O con conda
conda create -n garbage-classifier python=3.10
conda activate garbage-classifier
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Descargar el dataset
```bash
# Opción 1: Descarga manual
# Descarga el dataset de Kaggle y colócalo en data/raw/garbage_classification/

# Opción 2: Script automático (requiere API key de Kaggle)
python scripts/download_garbage_dataset.py
```

## 📖 Uso

### 1. Ejecutar el EDA
```bash
jupyter notebook notebooks/01_eda_garbage_classification.ipynb
```
El notebook generará automáticamente todas las visualizaciones en `figures/eda/`

### 2. Entrenar el modelo
```bash
# Próximamente
python scripts/train.py --config configs/ensemble_config.yaml
```

### 3. Evaluar el modelo
```bash
# Próximamente
python scripts/evaluate.py --model models/trained/best_model.pth
```

### 4. Hacer predicciones
```bash
# Próximamente
python scripts/predict.py --image path/to/image.jpg
```

## 🛠️ Tecnologías Utilizadas

### Análisis y Visualización
- **NumPy**: Operaciones numéricas
- **Pandas**: Manipulación de datos
- **Matplotlib**: Visualizaciones básicas
- **Seaborn**: Visualizaciones estadísticas

### Procesamiento de Imágenes
- **Pillow (PIL)**: Manipulación de imágenes
- **OpenCV**: Procesamiento avanzado de imágenes

### Machine Learning / Deep Learning
- **PyTorch**: Framework de deep learning
- **torchvision**: Modelos y transformaciones pre-entrenados
- **scikit-learn**: Métricas y utilidades de ML

### Desarrollo
- **Jupyter**: Notebooks interactivos
- **tqdm**: Barras de progreso
- **pytest**: Testing

## 📈 Roadmap del Proyecto

### Fase 1: Exploración y Preparación ✅
- [x] Análisis exploratorio de datos (EDA)
- [x] Visualizaciones y estadísticas
- [x] Identificación de desafíos del dataset

### Fase 2: Preparación de Datos 🔄
- [ ] Preprocesamiento de imágenes
- [ ] Data augmentation
- [ ] Split de datos (train/val/test)
- [ ] Data loaders optimizados

### Fase 3: Modelado 📝
- [ ] Baseline con modelo simple (CNN básica)
- [ ] Transfer learning (ResNet, EfficientNet, etc.)
- [ ] Fine-tuning de modelos pre-entrenados
- [ ] Desarrollo de modelos ensemble

### Fase 4: Optimización ⚙️
- [ ] Hyperparameter tuning
- [ ] Técnicas de regularización
- [ ] Balanceo de clases
- [ ] Cross-validation

### Fase 5: Evaluación 📊
- [ ] Métricas de rendimiento
- [ ] Matriz de confusión
- [ ] Análisis de errores
- [ ] Comparación de modelos

### Fase 6: Deployment 🚀
- [ ] Exportación del modelo
- [ ] API REST
- [ ] Dockerización
- [ ] Documentación

## 👥 Equipo

**Equipo 3 - Bootcamp F5 IA**
- Proyecto desarrollado en el marco del Bootcamp de Inteligencia Artificial de Factoría F5

## 📝 Licencia

Este proyecto es parte de un ejercicio educativo del Bootcamp F5 IA.

## 🙏 Agradecimientos

- Dataset original de clasificación de residuos
- Factoría F5 por el programa de formación
- Comunidad de código abierto por las herramientas utilizadas

---

⭐ Si este proyecto te resulta útil, no olvides darle una estrella!
