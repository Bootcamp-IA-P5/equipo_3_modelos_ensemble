from pathlib import Path
print(f'Total imágenes: {len(list(Path("data/raw/garbage_classification").rglob("*.jpg")))}')