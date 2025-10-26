"""
Genera label_map.json a partir de data/train.csv.
Salida: label_map.json con formato { "0": "cardboard", "1": "glass", ... }
"""
import json
from pathlib import Path
import pandas as pd

def build_label_map(train_csv: Path, out_json: Path):
    df = pd.read_csv(train_csv)
    if "filepath" not in df.columns or "label" not in df.columns:
        raise SystemExit("data/train.csv debe contener columnas 'filepath' y 'label'.")
    mapping = {}
    # Para cada label numérico tomamos la carpeta padre de la primera ruta encontrada
    for lbl, group in df.groupby("label"):
        first_path = group.iloc[0]["filepath"]
        # normalizar separadores y extraer carpeta padre
        p = Path(first_path.replace("\\", "/"))
        parent = p.parent.name
        mapping[int(lbl)] = parent
    out_json.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Saved label map to {out_json}")
    return mapping

if __name__ == "__main__":
    TRAIN = Path("data/train.csv")
    OUT = Path("label_map.json")
    if not TRAIN.exists():
        raise SystemExit("No se encontró data/train.csv. Ejecuta run_create_csv.py si hace falta.")
    build_label_map(TRAIN, OUT)