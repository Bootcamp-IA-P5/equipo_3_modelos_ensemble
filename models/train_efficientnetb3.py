"""
Script de entrenamiento para EfficientNetB3 usando el split en
`data/processed/garbage_classification_split/{train,val,test}/<class>/`.

Uso (ejemplo):
    python models/train_efficientnetb3.py --data_dir data/processed/garbage_classification_split \
        --epochs 10 --batch_size 32 --lr 1e-4 --output_dir models/trained

Guarda el modelo en formato HDF5 y el historial en JSON.
"""

import argparse
import os
from pathlib import Path
import json
from datetime import datetime
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento EfficientNetB3 - Transfer learning"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/garbage_classification_split",
        help="Ruta al directorio con train/val/test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/trained",
        help="Directorio donde guardar el modelo y el historial",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=(300, 300),
        help="Tamaño de entrada (h, w). EfficientNetB3 usa 300x300 por defecto",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--fine_tune_at",
        type=int,
        default=50,
        help="Número de capas para descongelar (desde el final) para fine-tuning",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Construir generadores y modelo, no entrenar",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Forzar uso de CPU (deshabilitar GPUs) antes de inicializar TensorFlow",
    )
    return parser.parse_args()


def build_generators(data_dir, img_size=(224, 224), batch_size=32, seed=42):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator


def build_model(num_classes, input_shape=(300, 300, 3), lr=1e-4, fine_tune_at=50):
    from tensorflow.keras import layers, models, optimizers

    try:
        # EfficientNetB3 está disponible en keras.applications
        from tensorflow.keras.applications import EfficientNetB3
    except Exception:
        # También puede estar disponible en tensorflow.keras.applications.efficientnet
        from tensorflow.keras.applications.efficientnet import EfficientNetB3

    # Intentar cargar pesos preentrenados; si hay mismatch de canales, hacer fallback a weights=None
    try:
        base_model = EfficientNetB3(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    except Exception as e:
        # Mismatch frecuente: modelo espera input con 1 canal pero los pesos son para 3 canales
        print("Warning: no se pudieron cargar los pesos 'imagenet' directamente:", e)
        print("Construyendo EfficientNetB3 sin pesos preentrenados (weights=None).")
        base_model = EfficientNetB3(
            weights=None, include_top=False, input_shape=input_shape
        )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if fine_tune_at and fine_tune_at > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False

    return model


def main():
    args = parse_args()
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf
    from tensorflow.keras import callbacks

    tf.random.set_seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen = build_generators(
        str(data_dir),
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    num_classes = len(train_gen.class_indices)

    try:
        model = build_model(
            num_classes,
            input_shape=(args.img_size[0], args.img_size[1], 3),
            lr=args.lr,
            fine_tune_at=args.fine_tune_at,
        )
    except Exception as e:
        print(
            "Error durante la construcción del modelo (posible problema con GPU/CUDA):",
            e,
        )
        if os.environ.get("TF_FALLBACK_TRIED") != "1":
            print("Reintentando en modo CPU (reiniciando el proceso)...")
            os.environ["TF_FALLBACK_TRIED"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = out_dir / f"efficientnetb3_{timestamp}.h5"
    history_path = out_dir / f"efficientnetb3_history_{timestamp}.json"

    cb_checkpoint = callbacks.ModelCheckpoint(
        str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1
    )
    cb_early = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    if args.dry_run:
        print("Dry run: generadores y modelo construidos correctamente")
        print("Clases (train):", train_gen.class_indices)
        print("Num clases:", num_classes)
        return

    try:
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=[cb_checkpoint, cb_early],
        )
    except Exception as e:
        print("Error durante el entrenamiento (posible problema con GPU/CUDA):", e)
        if os.environ.get("TF_FALLBACK_TRIED") != "1":
            print("Reintentando en modo CPU (reiniciando el proceso)...")
            os.environ["TF_FALLBACK_TRIED"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise

    hist = history.history
    with open(history_path, "w") as f:
        json.dump(hist, f)

    print("Entrenamiento finalizado. Modelo guardado en:", model_path)
    print("Historial guardado en:", history_path)


if __name__ == "__main__":
    main()
