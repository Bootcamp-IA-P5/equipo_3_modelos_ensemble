Esta carpeta contiene utilidades y scripts “one‑off” usados para preparar datos, extraer features,
ejecutar tuning ligero y realizar inferencia.

Archivos principales (descripción breve):
- run_create_csv.py            -> crea data/train.csv a partir de processed/splits.json o raw/
- run_fix_splits_to_csv.py     -> arregla formatos específicos de splits.json y genera data/train.csv
- run_images_to_features.py    -> extrae features (64x64 aplanado) y guarda data/features_small.csv
- run_optuna_test.py           -> script de prueba para lanzar optimize_lightgbm_light
- save_best_params.py          -> guarda manualmente optuna_best_params.json (si se requiere)
- temp_run_features.py         -> script temporal para testing de features + tuning
- temp_count_images.py         -> script auxiliar para contar imágenes
- test_optuna_integration.py   -> pruebas unitarias/integ para Optuna integration
- train_final_model.py         -> reentrena el modelo final y guarda models/lightgbm_final.joblib
- predict_example.py           -> script de inferencia de ejemplo
- map_labels.py                -> genera label_map.json a partir de data/train.csv

Cómo usar
- Ejecuta los scripts desde la raíz del repositorio (esto mantiene rutas relativas a `data/` funcionando):
  cd <repo-root>
  python scripts/run_images_to_features.py

Notas
- Si ejecutas desde otra carpeta, añade `sys.path.insert(0, os.path.abspath("."))` al inicio de los scripts que importen desde `src/`.
- No commitear modelos grandes: usa `models/` en .gitignore o sube el joblib a storage externo.
```

$files = @(
  "run_create_csv.py",
  "run_fix_splits_to_csv.py",
  "run_images_to_features.py",
  "run_optuna_test.py",
  "save_best_params.py",
  "temp_count_images.py",
  "temp_run_features.py",
  "test_optuna_integration.py",
  "train_final_model.py",
  "predict_example.py",
  "predict.py",
  "map_labels.py"
)
foreach ($f in $files) {
    if (Test-Path $f) {
        Move-Item -Path $f -Destination .\scripts\ -Force
        Write-Host "Moved:" $f
    } else {
        Write-Host "No encontrado (skip):" $f
    }
}