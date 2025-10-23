"""
Ensamble de modelos para clasificación de basura.

Soporta Voting (soft/hard) y Bagging (promedio de probabilidades) sobre modelos
preentrenados guardados en HDF5. También puede construirse en modo dry-run para
verificar generadores y la disponibilidad de pesos.

Uso (ejemplo):
    python models/ensemble.py --model_paths models/trained/resnet50_xxx.h5 \
        models/trained/vgg16_xxx.h5 --method voting_soft --data_dir data/... --img_size 224 224

"""

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memoria GPU en modo dinámico")
    except RuntimeError as e:
        print(e)


import argparse
import os
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import accuracy_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Ensamble de modelos (voting/bagging)")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="*",
        default=[],
        help="Rutas a los archivos .h5 de los modelos a incluir en el ensamble",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["voting_soft", "voting_hard", "bagging"],
        default="voting_soft",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/garbage_classification_split",
    )
    parser.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_generators(data_dir, img_size=(224, 224), batch_size=32):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return val_generator


def load_models(model_paths):
    from tensorflow.keras.models import load_model

    models = []
    for p in model_paths:
        p = Path(p)
        if not p.exists():
            print(f"Warning: modelo no encontrado: {p} -> será ignorado")
            continue
        try:
            # compile=False evita necesidad de deserializar optimizadores/métricas
            m = load_model(str(p), compile=False)
            models.append((p.name, m))
            print(f"Cargado: {p}")
        except Exception as e:
            print(f"Error cargando {p}: {e}")
    return models


def predict_ensemble(models, generator, method="voting_soft"):
    # models: list of (name, model)
    n = len(models)
    if n == 0:
        raise ValueError("No hay modelos cargados para el ensamble")

    steps = int(np.ceil(generator.samples / generator.batch_size))
    # Obtener predicciones de cada modelo
    all_preds = []
    for name, m in models:
        preds = m.predict(generator, steps=steps, verbose=0)
        all_preds.append(preds)

    all_preds = np.array(all_preds)  # (n_models, n_samples, n_classes)

    if method == "voting_soft" or method == "bagging":
        avg = np.mean(all_preds, axis=0)
        y_pred = np.argmax(avg, axis=1)
    elif method == "voting_hard":
        # convertir a etiquetas por modelo y votar
        votes = np.argmax(all_preds, axis=2)  # (n_models, n_samples)
        # mayoría simple
        from scipy.stats import mode

        mode_res = mode(votes, axis=0)
        y_pred = mode_res.mode[0]
    else:
        raise ValueError("Método desconocido")

    return y_pred


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    val_gen = build_generators(
        str(data_dir), img_size=tuple(args.img_size), batch_size=args.batch_size
    )

    if args.dry_run:
        print("Dry run: generador de validación creado")
        print("Num muestras val:", val_gen.samples)
        print("Clases:", val_gen.class_indices)
        return

    # Si no se pasan rutas, intentamos descubrir automáticamente modelos en models/trained/*.h5
    model_paths = args.model_paths
    if not model_paths:
        default_glob = repo_root / "models" / "trained" / "*.h5"
        discovered = list(map(str, Path(repo_root / "models" / "trained").glob("*.h5")))
        if discovered:
            print("No se proporcionaron --model_paths. Usando modelos detectados:")
            for p in discovered:
                print(" -", p)
            model_paths = discovered
        else:
            print(
                "No se encontraron .h5 en models/trained/. Puedes especificarlos con --model_paths"
            )

    models = load_models(model_paths)
    if len(models) == 0:
        print("No hay modelos disponibles para el ensamble. Salir.")
        return

    y_true = val_gen.classes

    y_pred = predict_ensemble(models, val_gen, method=args.method)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Ensamble method={args.method} - accuracy={acc:.4f}")
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
