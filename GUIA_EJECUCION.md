# 🚀 Guía Completa de Ejecución - EcoSort

## ✅ Checklist Previo

Antes de empezar, asegúrate de tener:

- [ ] Python 3.10 o superior instalado
- [ ] 10GB de espacio libre en disco
- [ ] GPU con 6GB VRAM (opcional pero recomendado)
- [ ] Dataset de garbage classification descargado

---

## 📦 Paso 1: Instalación

### 1.1 Clonar el Repositorio

```bash
git clone https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble.git
cd equipo_3_modelos_ensemble
```

### 1.2 Crear Entorno Virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 1.4 Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota para GPU:** Si tienes GPU NVIDIA con CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📂 Paso 2: Preparar el Dataset

### 2.1 Estructura Necesaria

```
data/raw/garbage_classification/
├── cardboard/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

### 2.2 Verificar Dataset

```bash
python -c "from pathlib import Path; print(f'Total imágenes: {len(list(Path(\"data/raw/garbage_classification\").rglob(\"*.jpg\")))}')"
```

Deberías ver aproximadamente 2,500+ imágenes.

---

## 🏋️ Paso 3: Entrenar los Modelos

### 3.1 Entrenamiento Básico (Rápido)

```bash
python scripts/train_models.py \
  --epochs 20 \
  --batch_size 32 \
  --img_size 128 \
  --model_name resnet18
```

⏱️ Tiempo estimado:
- Con GPU: 30-45 minutos
- Con CPU: 2-3 horas

### 3.2 Entrenamiento Completo (Recomendado)

```bash
python scripts/train_models.py \
  --epochs 30 \
  --batch_size 32 \
  --img_size 128 \
  --model_name resnet18 \
  --optimize_hp \
  --hp_trials 30
```

⏱️ Tiempo estimado:
- Con GPU: 2-3 horas
- Con CPU: 6-8 horas

### 3.3 Configuraciones Alternativas

#### Para 4GB VRAM (GPU más pequeña):
```bash
python scripts/train_models.py \
  --epochs 30 \
  --batch_size 16 \
  --img_size 96 \
  --model_name mobilenet_v2
```

#### Para 8GB+ VRAM (GPU más grande):
```bash
python scripts/train_models.py \
  --epochs 40 \
  --batch_size 64 \
  --img_size 224 \
  --model_name resnet18
```

### 3.4 Verificar Modelos Entrenados

```bash
ls models/trained/exp_*/
```

Deberías ver:
- `best_model.pth` - Modelo CNN
- `stacking_model.pkl` - Modelo ensemble (recomendado)
- `voting_model.pkl` - Alternativa
- `final_results.json` - Métricas

---

## 🎨 Paso 4: Lanzar la Aplicación Web

### 4.1 Ejecutar Streamlit

```bash
streamlit run app/app_streamlit.py
```

### 4.2 Configurar Rutas de Modelos

Si los modelos no se cargan automáticamente, edita en `app/app_streamlit.py`:

```python
CNN_MODEL_PATH = "models/trained/exp_YYYYMMDD_HHMMSS/best_model.pth"
ENSEMBLE_MODEL_PATH = "models/trained/exp_YYYYMMDD_HHMMSS/stacking_model.pkl"
```

### 4.3 Acceder a la Aplicación

1. Abre tu navegador en: **http://localhost:8501**
2. Sube una imagen de prueba
3. Haz clic en "Clasificar Residuo"
4. ¡Disfruta de la predicción!

---

## 🧪 Paso 5: Probar el Sistema

### 5.1 Predicción desde CLI

```bash
python scripts/predict.py \
  --image data/raw/garbage_classification/cardboard/cardboard1.jpg \
  --cnn_model models/trained/exp_*/best_model.pth \
  --ensemble_model models/trained/exp_*/stacking_model.pkl
```

### 5.2 Verificar Salida

Deberías ver algo como:

```json
{
  "prediction": "cardboard",
  "confidence": 0.9543,
  "probabilities": {
    "cardboard": 0.9543,
    "glass": 0.0234,
    "metal": 0.0089,
    "paper": 0.0078,
    "plastic": 0.0045,
    "trash": 0.0011
  },
  "recycling_info": {
    "name": "Cartón",
    "bin": "📦 Contenedor Azul",
    "tips": "Asegúrate de que esté limpio y plegado."
  }
}
```

---

## 🐳 Paso 6: Deployment con Docker (Opcional)

### 6.1 Construir Imagen

```bash
docker build -t ecosort:latest .
```

### 6.2 Ejecutar Contenedor

```bash
docker run -p 8501:8501 \
  -v $(pwd)/models/trained:/app/models/trained:ro \
  -v $(pwd)/data/feedback:/app/data/feedback \
  ecosort:latest
```

### 6.3 O usar Docker Compose

```bash
docker-compose up -d
```

### 6.4 Verificar Estado

```bash
docker-compose ps
docker-compose logs -f
```

### 6.5 Detener

```bash
docker-compose down
```

---

## 📊 Paso 7: Ver Métricas y Resultados

### 7.1 Resultados del Entrenamiento

```bash
cat models/trained/exp_*/final_results.json
```

### 7.2 Curvas de Entrenamiento

Las gráficas se generan automáticamente:
- `models/trained/exp_*/cnn_training_curves.png`

### 7.3 Métricas en Tiempo Real

1. Abre la aplicación web
2. Ve a "📊 Métricas" en el sidebar
3. Visualiza:
   - Accuracy actual
   - Distribución de predicciones
   - Feedback recibido

---

## 🔄 Paso 8: Sistema de Feedback

### 8.1 Recopilar Feedback

1. Clasifica imágenes en la aplicación web
2. Marca predicciones como correctas/incorrectas
3. Añade comentarios si deseas

### 8.2 Ver Feedback Almacenado

```bash
ls data/feedback/
cat data/feedback/feedback_*.json
```

### 8.3 Exportar para Reentrenamiento

En la aplicación web:
1. Ve a "📝 Feedback"
2. Filtra datos si necesario
3. Haz clic en "📥 Exportar datos para reentrenamiento"

---

## 🐛 Solución de Problemas

### Error: CUDA out of memory

```bash
# Reducir batch size
python scripts/train_models.py --batch_size 16 --img_size 96
```

### Error: ModuleNotFoundError

```bash
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: Modelo no encuentra archivo

```bash
# Verificar rutas
ls -la models/trained/exp_*/

# Usar ruta absoluta
python scripts/predict.py --cnn_model /ruta/completa/best_model.pth
```

### Aplicación Streamlit no carga

```bash
# Verificar puerto
streamlit run app/app_streamlit.py --server.port=8502

# Ver logs
streamlit run app/app_streamlit.py --logger.level=debug
```

### Error de permisos en Docker

```bash
# Linux/Mac: Añadir permisos
chmod -R 755 models/ data/

# Windows: Ejecutar como administrador
```

---

## ✨ Consejos y Mejores Prácticas

### Para Mejor Rendimiento

1. **Usa GPU si es posible**: 10-20x más rápido
2. **Ajusta batch_size**: Según tu VRAM disponible
3. **Monitorea recursos**: `nvidia-smi` (GPU) o `htop` (CPU)

### Para Mejores Resultados

1. **Entrena más épocas**: 40-50 para mejor convergencia
2. **Optimiza hiperparámetros**: Usa `--optimize_hp`
3. **Aumenta dataset**: Recopila más imágenes con feedback

### Para Desarrollo

1. **Usa entorno virtual**: Aísla dependencias
2. **Versiona modelos**: Guarda experimentos con timestamps
3. **Documenta cambios**: Mantén CHANGELOG.md

---

## 📞 Soporte y Ayuda

### Documentación Adicional

- 📖 [README.md](README.md) - Documentación completa
- 🚀 [QUICKSTART.md](QUICKSTART.md) - Inicio rápido
- 📊 [INFORME_TECNICO.md](reports/INFORME_TECNICO.md) - Análisis técnico

### Contacto

- 🐛 Issues: [GitHub Issues](https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble/issues)
- 📧 Email: equipo3@bootcamp-f5.com
- 💬 Discussions: [GitHub Discussions](https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble/discussions)

---

## ✅ Checklist Final

Después de seguir todos los pasos, deberías tener:

- [x] Entorno configurado
- [x] Dataset preparado
- [x] Modelos entrenados
- [x] Aplicación web funcionando
- [x] Predicciones realizadas
- [x] Sistema de feedback activo

---

## 🎯 Próximos Pasos

1. **Recopila más datos** mediante el sistema de feedback
2. **Reentrena modelos** con datos nuevos cada semana
3. **Monitorea métricas** para detectar degradación
4. **Optimiza inferencia** para producción
5. **Comparte tu experiencia** con la comunidad

---

**¡Felicidades! 🎉 Has completado la configuración de EcoSort**

♻️ **EcoSort** - Equipo 3, Bootcamp F5 IA | 2024
