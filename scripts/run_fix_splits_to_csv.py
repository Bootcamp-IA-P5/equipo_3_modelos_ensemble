import json
from pathlib import Path
import csv
import sys

ROOT = Path.cwd()
SPLITS = ROOT / "data" / "processed" / "splits.json"
OUT = ROOT / "data" / "train.csv"

if not SPLITS.exists():
    print(f"Error: {SPLITS} no encontrado.")
    sys.exit(1)

with open(SPLITS, "r", encoding="utf-8") as f:
    splits = json.load(f)

if "train" not in splits:
    print("El split 'train' no se encontró en splits.json. Claves top-level:", list(splits.keys()))
    sys.exit(1)

train = splits["train"]

rows = []
# Caso 1: train es lista de strings
if isinstance(train, list) and all(isinstance(x, str) for x in train):
    for p in train:
        label = Path(p).parent.name
        rows.append({"filepath": p, "label": label})
# Caso 2: train es dict con 'paths' y 'labels' arrays paralelos
elif isinstance(train, dict) and "paths" in train and "labels" in train:
    paths = train["paths"]
    labels = train["labels"]
    if len(paths) != len(labels):
        print(f"Advertencia: paths ({len(paths)}) y labels ({len(labels)}) tienen distinto largo; usando min length.")
    n = min(len(paths), len(labels))
    for i in range(n):
        rows.append({"filepath": paths[i], "label": labels[i]})
# Caso 3: train es lista de dicts con keys file/path & label/class
elif isinstance(train, list) and all(isinstance(x, dict) for x in train):
    for it in train:
        file_key = None
        label_key = None
        for k in ("file", "filepath", "path", "image", "img", "pathfile"):
            if k in it:
                file_key = k
                break
        for k in ("label", "class", "category", "target"):
            if k in it:
                label_key = k
                break
        if file_key is None:
            # intentar detectar primer string que parezca path
            for k,v in it.items():
                if isinstance(v, str) and (v.endswith(".jpg") or v.endswith(".png") or "/" in v or "\\" in v):
                    file_key = k
                    break
        if file_key is None:
            print("Skipping entry (no file key):", it)
            continue
        file_val = it[file_key]
        label = it[label_key] if label_key and label_key in it else Path(file_val).parent.name
        rows.append({"filepath": file_val, "label": label})
else:
    print("Formato inesperado para 'train'. Tipo:", type(train))
    sys.exit(1)

if not rows:
    print("No se generaron filas. Revisa la estructura de splits.json")
    sys.exit(1)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Creado {OUT} con {len(rows)} filas.")