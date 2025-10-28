"""Crea data/train.csv a partir de:
 - processed/splits.json (si existe), o
 - raw/garbage_classification/* (cada subcarpeta = etiqueta).

Salida: data/train.csv con columnas: filepath,label
Rutas en CSV serán relativas a la raíz del repo (p. ej. raw/garbage_classification/plastic/img1.jpg).
"""
import json
import os
from pathlib import Path
import csv

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
PROCESSED_SPLITS = DATA_DIR / "processed" / "splits.json"
RAW_DIR = DATA_DIR / "raw"

def create_csv_from_splits(splits_path: Path, out_csv: Path, split_name: str = "train") -> bool:
    try:
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
    except Exception as e:
        print(f"Error leyendo {splits_path}: {e}")
        return False

    if isinstance(splits, dict) and split_name in splits:
        items = splits[split_name]
        rows = []
        for it in items:
            if isinstance(it, str):
                path = Path(it)
                label = path.parent.name
                rows.append({"filepath": str(it), "label": label})
            elif isinstance(it, dict):
                file_key = None
                label_key = None
                for k in ("file", "filepath", "path", "image", "img"):
                    if k in it:
                        file_key = k
                        break
                for k in ("label", "class", "category", "target"):
                    if k in it:
                        label_key = k
                        break
                if file_key is None:
                    for k, v in it.items():
                        if isinstance(v, str) and (v.endswith(".jpg") or v.endswith(".png") or "/" in v):
                            file_key = k
                            break
                if file_key is None:
                    print("No se encontró clave de archivo en entrada:", it)
                    continue
                file_val = it[file_key]
                label = it[label_key] if label_key else Path(file_val).parent.name
                rows.append({"filepath": str(file_val), "label": label})
        if rows:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"Creado {out_csv} ({len(rows)} filas) desde split '{split_name}' en {splits_path}")
            return True
    print(f"No se pudo crear CSV desde {splits_path} para split '{split_name}'.")
    return False

def create_csv_from_raw(raw_dir: Path, out_csv: Path) -> bool:
    gc_dir = raw_dir / "garbage_classification"
    if not gc_dir.exists():
        print(f"{gc_dir} no existe.")
        return False
    rows = []
    for label_dir in sorted(gc_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    rel = Path(root) / file
                    rows.append({"filepath": str(rel.as_posix()), "label": label})
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Creado {out_csv} ({len(rows)} filas) desde {gc_dir}")
        return True
    print("No se encontraron imágenes bajo", gc_dir)
    return False

def main():
    out_csv = DATA_DIR / "train.csv"
    if PROCESSED_SPLITS.exists():
        print("Encontrado processed/splits.json -> intentando crear CSV desde 'train' split.")
        ok = create_csv_from_splits(PROCESSED_SPLITS, out_csv, split_name="train")
        if ok:
            return
        else:
            print("Fallo al crear desde splits.json; intentar crear desde raw.")
    ok2 = create_csv_from_raw(RAW_DIR, out_csv)
    if ok2:
        return
    print("No se pudo crear data/train.csv automáticamente.")
    feedback_dir = DATA_DIR / "feedback"
    if feedback_dir.exists():
        print("JSONs de feedback encontrados en data/feedback:")
        for j in feedback_dir.glob("*.json"):
            print("  -", j)
    else:
        print("No se encontraron JSONs de feedback. Considera crear un CSV sintético o colocar tu CSV en data/train.csv")

if __name__ == "__main__":
    main()
