import json
from pathlib import Path
import argparse
import sys

import pandas as pd
from lightgbm import LGBMClassifier
from joblib import dump
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

# Defaults (ajusta si quieres)
DEFAULT_FEATURES = Path("data/features_small.csv")   # Cambia a data/features_all.csv si lo generas
DEFAULT_PARAMS = Path("optuna_best_params.json")
DEFAULT_OUT = Path("models/lightgbm_final.joblib")
DEFAULT_CV_FOLDS = 5


def load_best_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Params JSON no encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", {}) if isinstance(payload, dict) else {}
    # Asegurar tipos enteros donde corresponda
    for k in ("n_estimators", "max_depth", "num_leaves"):
        if k in params:
            try:
                params[k] = int(params[k])
            except Exception:
                pass
    # Defaults seguros
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", -1)
    return params


def main(args):
    features_csv = Path(args.features)
    params_json = Path(args.params)
    out_model = Path(args.out)
    cv_folds = args.cv_folds

    if not features_csv.exists():
        print(f"ERROR: Features CSV no encontrado: {features_csv}", file=sys.stderr)
        print(" - Si aún no has creado features para todas las imágenes, ejecuta run_images_to_features.py o crea data/features_all.csv", file=sys.stderr)
        sys.exit(1)

    if not params_json.exists():
        print(f"ERROR: Params JSON no encontrado: {params_json}", file=sys.stderr)
        print(" - Asegúrate de que optuna_best_params.json exista en la raíz del repo", file=sys.stderr)
        sys.exit(1)

    print("Cargando features desde:", features_csv)
    df = pd.read_csv(features_csv)
    if "target" not in df.columns:
        print("ERROR: El CSV de features debe contener la columna 'target' con las etiquetas.", file=sys.stderr)
        sys.exit(1)

    X = df.drop(columns=["target"])
    y = df["target"]

    print("Cargando mejores parámetros desde:", params_json)
    params = load_best_params(params_json)
    print("Parámetros utilizados (post-proc):", params)

    print(f"Entrenando LGBMClassifier con {X.shape[0]} muestras y {X.shape[1]} features...")
    clf = LGBMClassifier(**params)

    # Evaluación opcional por CV (rápida comprobación)
    try:
        print(f"Validando con cross-val ({cv_folds} folds) usando F1 macro...")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=params.get("random_state", 42))
        scorer = make_scorer(f1_score, average="macro")
        scores = cross_val_score(clf, X, y, cv=skf, scoring=scorer, n_jobs=-1)
        print(f"  CV F1 macro mean: {scores.mean():.4f}  std: {scores.std():.4f}")
    except Exception as e:
        print("Advertencia: no se pudo ejecutar cross_val_score (continuando con fit). Error:", e)

    # Fit final sobre todo el dataset
    clf.fit(X, y)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, out_model)
    print("Modelo entrenado y guardado en:", out_model)
    print("Listo. Para inferencia carga el joblib y aplica transformaciones idénticas a las usadas para generar el CSV de features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo final LightGBM usando params de Optuna y un CSV de features.")
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATURES),
                        help="Ruta al CSV de features (default: data/features_small.csv). Debe contener columna 'target'.")
    parser.add_argument("--params", type=str, default=str(DEFAULT_PARAMS),
                        help="Ruta al JSON con best params (default: optuna_best_params.json).")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Ruta de salida para el modelo serializado (joblib).")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS,
                        help="Número de folds para evaluación rápida antes de entrenar (default: 5).")
    args = parser.parse_args()
    main(args)