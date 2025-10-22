#!/bin/bash
# Script de prueba rápida del modelo

cd "$(dirname "$0")"

echo "======================================"
echo "PRUEBA RÁPIDA DEL MODELO"
echo "======================================"
echo

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source .venv/Scripts/activate

# Probar predict.py
echo "🧪 Probando predict.py..."
python scripts/predict.py \
    --image "data/raw/garbage_classification/cardboard/cardboard1.jpg" \
    --cnn_model "models/trained/exp_20251022_110227/best_model.pth" \
    --ensemble_model "models/trained/exp_20251022_110227/stacking_model.pkl"

echo
echo "======================================"
echo "✅ Prueba completada"
echo "======================================"
