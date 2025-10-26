import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

CSV = Path("data/train.csv")
OUT = Path("data/features_small.csv")
N_SAMPLES = 200        # ajusta si quieres más/menos
IMG_SIZE = (64, 64)    # tamaño al que redimensionamos las imágenes

def main():
    if not CSV.exists():
        raise SystemExit(f"{CSV} no existe. Ejecuta run_create_csv.py primero.")
    df = pd.read_csv(CSV)
    # Filtrar solo filas con archivo existente
    df['exists'] = df['filepath'].apply(lambda p: Path(p).exists())
    df = df[df['exists']].copy()
    if df.empty:
        raise SystemExit("No se encontraron archivos de imagen existentes para las rutas en data/train.csv")
    # Mezclar y limitar
    df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(N_SAMPLES)

    features = []
    labels = []
    for _, row in df.iterrows():
        p = Path(row['filepath'])
        try:
            img = Image.open(p).convert('L').resize(IMG_SIZE)
            arr = np.asarray(img, dtype=np.float32).ravel() / 255.0
            features.append(arr)
            labels.append(row['label'])
        except Exception as e:
            print("skip", p, e)

    if not features:
        raise SystemExit("No se pudo extraer ninguna feature de las imágenes seleccionadas.")

    X = np.vstack(features)
    y = np.array(labels)

    cols = [f"f_{i}" for i in range(X.shape[1])]
    out_df = pd.DataFrame(X, columns=cols)
    out_df['target'] = y
    out_df.to_csv(OUT, index=False)
    print("Saved features to", OUT, "shape:", out_df.shape)

if __name__ == "__main__":
    main()