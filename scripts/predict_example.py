import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from joblib import load
import pandas as pd
import json

IMG_SIZE = (64, 64)   # Debe coincidir con cómo generaste las features
FEATURE_COL_PREFIX = "f_"

def image_to_feature_vector(image_path: Path):
    img = Image.open(image_path).convert("L").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32).ravel() / 255.0
    return arr

def vec_to_dataframe(vec: np.ndarray):
    cols = [f"{FEATURE_COL_PREFIX}{i}" for i in range(vec.shape[0])]
    return pd.DataFrame([vec], columns=cols)

def load_label_map(path: Path):
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    return None

def predict_from_image(model_path: Path, image_path: Path, label_map=None):
    model = load(model_path)
    vec = image_to_feature_vector(image_path)
    df = vec_to_dataframe(vec)
    pred = model.predict(df)
    proba = model.predict_proba(df)
    label = label_map.get(int(pred[0])) if label_map else None
    return int(pred[0]), proba[0].tolist(), label

def predict_from_features_csv(model_path: Path, features_csv: Path, index: int = 0, label_map=None):
    model = load(model_path)
    df = pd.read_csv(features_csv)
    if "target" in df.columns:
        X = df.drop(columns=["target"])
    else:
        X = df
    row = X.iloc[[index]]
    pred = model.predict(row)
    proba = model.predict_proba(row)
    label = label_map.get(int(pred[0])) if label_map else None
    return int(pred[0]), proba[0].tolist(), label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejemplo de inferencia con el modelo lightgbm_final.joblib")
    parser.add_argument("--model", type=str, default="models/lightgbm_final.joblib")
    parser.add_argument("--image", type=str, help="Ruta a una imagen para predecir (uso único)")
    parser.add_argument("--features_csv", type=str, help="Ruta a CSV de features (usa --index para elegir fila)")
    parser.add_argument("--index", type=int, default=0, help="Índice de la fila en features_csv si se usa features_csv")
    parser.add_argument("--label-map", type=str, default="label_map.json", help="Ruta al JSON con mapping num->label")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Modelo no encontrado: {model_path}")

    label_map = load_label_map(Path(args.label_map))

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise SystemExit(f"Imagen no encontrada: {img_path}")
        pred_idx, proba, label = predict_from_image(model_path, img_path, label_map)
        print("Prediction index:", pred_idx)
        if label is not None:
            print("Prediction label:", label)
        print("Probabilities:", proba)
    elif args.features_csv:
        csv_path = Path(args.features_csv)
        if not csv_path.exists():
            raise SystemExit(f"CSV de features no encontrado: {csv_path}")
        pred_idx, proba, label = predict_from_features_csv(model_path, csv_path, index=args.index, label_map=label_map)
        print(f"Prediction index for row {args.index}:", pred_idx)
        if label is not None:
            print("Prediction label:", label)
        print("Probabilities:", proba)
    else:
        raise SystemExit("Proporciona --image o --features_csv.")