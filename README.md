# 🗑️ Clasificación de Residuos con Modelos Ensemble

## 📋 Tabla de Contenidos
- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Contexto de Negocio](#-contexto-de-negocio)
- [Dataset](#-dataset)
- [Análisis Exploratorio de Datos (EDA)](#-análisis-exploratorio-de-datos-eda)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Equipo](#-equipo)

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
