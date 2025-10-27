# scripts/ — Quickstart (versión ligera)

Esta carpeta contiene utilidades y scripts "one‑off" para preparar datos, extraer features,
realizar tuning ligero con Optuna y realizar inferencia. Este README explica el flujo
mínimo para que cualquier persona pueda ejecutar la "versión ligera" del pipeline.

Resumen de objetivos (versión ligera)
- Extraer features simples (grayscale 64x64 aplanado) para prototipado rápido.
- Ejecutar un tuning ligero de LightGBM (Optuna) con persistencia en SQLite y pruning.
- Entrenar un modelo final simple y guardar el joblib.
- Ejecutar inferencia con el modelo guardado.

Estructura importante
- scripts/
  - run_create_csv.py        -> genera data/train.csv (mapa filepath + label)
  - run_fix_splits_to_csv.py -> transforma splits a CSV si es necesario
  - run_images_to_features.py-> extrae features simples -> data/features_small.csv
  - run_optuna_test.py       -> ejemplo de ejecución de Optuna (tuning ligero)
  - save_best_params.py      -> guardar params resultantes de Optuna
  - temp_* / test_*          -> utilitarios y pruebas
  - train_final_model.py     -> reentrena y guarda models/lightgbm_final.joblib
  - predict_example.py       -> ejemplo de inferencia desde imagen o CSV
  - map_labels.py            -> genera label_map.json (num -> nombre clase)

Requisitos mínimos (versión ligera)
- Python 3.8+
- Virtualenv (recomendado)
- Dependencias del proyecto (requirements.txt)

Quickstart (Windows / PowerShell)
1) Desde la raíz del repo, crear y activar entorno:
   python -m venv .venv
   .\.venv\Scripts\Activate

2) Instalar dependencias:
   pip install -r requirements.txt

3) Ejecutar el pipeline ligero (ejemplo paso a paso):
   # 1. Generar data/train.csv (si no existe)
   python scripts\run_create_csv.py

   # 2. Extraer un subconjunto de features (rápido)
   python scripts\run_images_to_features.py
   # Esto crea: data/features_small.csv

   # 3. (Opcional) Ejecutar un tuning ligero con Optuna (puede tardar minutos)
   python scripts\run_optuna_test.py

   # 4. Guardar mejores parámetros si quieres versionarlos
   python scripts\save_best_params.py

   # 5. Entrenar modelo final (usa data/features_small.csv y optuna_best_params.json por defecto)
   python scripts\train_final_model.py

   # 6. Inferir sobre una imagen de ejemplo
   python scripts\predict_example.py --image data/raw/garbage_classification/plastic/plastic278.jpg

Quickstart (Unix / macOS)
1) python3 -m venv .venv
   source .venv/bin/activate
2) pip install -r requirements.txt
3) Usa las mismas invocaciones anteriores, reemplazando las barras por '/'.

Notas sobre preprocesado y reproducibilidad
- Asegúrate de ejecutar los scripts desde la raíz del repositorio (pwd = repo root). Los scripts usan rutas relativas como data/...
- El pipeline ligero usa:
  - Conversión a escala de grises
  - Redimensionado a 64x64 (IMG_SIZE = (64,64))
  - Aplanado y normalizado (/255.0)
- Si en algún momento generas features diferentes (embeddings CNN, etc.), debes reproducir exactamente las mismas transformaciones en inferencia.

Archivos importantes y recomendaciones de versionado
- Guardar en el repo:
  - optuna_best_params.json (pequeño, opcional)
  - label_map.json (útil para inferencia legible)
- NO commitear:
  - models/ (modelos pesados). Recomendado: añadir `models/` a .gitignore y subir modelos grandes a storage externo (S3, Drive).
  - data/features*.csv y optuna_lightgbm.db (artefactos generados). Manténlos fuera del historial.
- .gitattributes:
  - Incluido para normalizar finales de línea (LF) en archivos de texto y evitar diffs entre plataformas.

Troubleshooting rápido
- Warning "X does not have valid feature names" en predict: crea un DataFrame con columnas llamadas f_0..f_N-1 antes de predecir (el script predict_example.py ya lo hace).
- Warnings CRLF/LF en Windows: ejecutar `git config --global core.autocrlf true` y mantener .gitattributes.
- ImportError al ejecutar scripts: ejecuta desde la raíz del repo o añade al inicio del script:
  import sys, os
  sys.path.insert(0, os.path.abspath("."))

Checklist antes de abrir PR (para esta rama)
- [ ] Ejecutar smoke test: python scripts/predict_example.py --image <ruta>
- [ ] Verificar que .gitignore excluye models/ y artefactos generados
- [ ] No subir archivos grandes (revisar git status / git show --name-only HEAD)
- [ ] Comprobar que label_map.json y optuna_best_params.json están incluidos sólo si se quiso versionar

Contacto / autor
- Rama: feature/organize-scripts
- Autor de la organización: equipo_3_modelos_ensemble
- Si algo falla en la ejecución, copia la salida completa y pégala en la PR como un comentario para facilitar la reproducción.

Gracias — este README está pensado para que cualquiera (incluso sin contexto previo) pueda ejecutar la versión ligera del pipeline y obtener resultados reproducibles rápidos.
