import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns

# ---------------------------
# CONFIGURACIÓN PARA ALTA PRECISIÓN + OVERFITTING < 5%
# ---------------------------
DATA_DIR = Path('../data/raw/garbage_classification')
DATA_PATH = DATA_DIR 
REPORTS_DIR = Path('../reports/reports_cnn')
MODELS_DIR = Path('../reports/models')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15 # Más épocas para convergencia lenta pero estable
TRAIN_RATIO = 0.8
SEED = 123

# Crear directorios de salida
for d in [REPORTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("🎯 OBJETIVO: Alta precisión + Overfitting < 5%")

print("Cargando dataset desde directorio...")
full_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_PATH,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False  
)
class_names = full_ds.class_names
num_classes = len(class_names)
print(f"Clases detectadas: {class_names} (n={num_classes})")

# Crear división balanceada 80/20
def create_balanced_split(dataset, train_ratio=0.8):
    all_images = []
    all_labels = []
    for images, labels in dataset:
        all_images.extend(images.numpy())
        all_labels.extend(labels.numpy())
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        class_indices[label].append(idx)
    train_indices = []
    val_indices = []
    for class_idx, indices in class_indices.items():
        n_train = int(len(indices) * train_ratio)
        np.random.seed(SEED)
        np.random.shuffle(indices)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])
        print(f"  {class_names[class_idx]}: {n_train} train, {len(indices)-n_train} val")
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    train_images = all_images[train_indices]
    train_labels = all_labels[train_indices]
    val_images = all_images[val_indices]
    val_labels = all_labels[val_indices]
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

train_dataset, val_dataset = create_balanced_split(full_ds, train_ratio=TRAIN_RATIO)
print("Dataset dividido y listo.")

# ---------------------------
# DATA AUGMENTATION MÁS AGRESIVO
# ---------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.GaussianNoise(0.1),
])

def preprocess(image, label):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# Aplicar augmentation solo al training
train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                 num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# ---------------------------
# MODELO BALANCEADO: PRECISIÓN + BAJO OVERFITTING
# ---------------------------
print("Construyendo modelo balanceado...")

try:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    print("✅ MobileNetV2 cargado exitosamente con pesos de ImageNet")
except:
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    print("✅ MobileNetV2 con pesos aleatorios")

# ESTRATEGIA INTELIGENTE:
# - Congelar más capas al inicio, luego fine-tuning progresivo
base_model.trainable = True

# Congelar diferentes cantidades de capas basado en épocas
class ProgressiveUnfreezing(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 10:
            # Fase 1: Solo últimas capas
            for layer in base_model.layers[:-80]:
                layer.trainable = False
        elif epoch < 25:
            # Fase 2: Más capas
            for layer in base_model.layers[:-50]:
                layer.trainable = False
        else:
            # Fase 3: Todas las capas con bajo LR
            for layer in base_model.layers:
                layer.trainable = True

print(f"🔧 Fine-tuning progresivo activado")

# ---------------------------
# ARQUITECTURA OPTIMIZADA
# ---------------------------
print("Construyendo modelo completo optimizado...")

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)

# ARQUITECTURA BALANCEADA:
x = Dropout(0.5)(x)  # Dropout moderado
x = Dense(512, activation='relu')(x)  # Más capacidad
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)


initial_learning_rate = 1e-4

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"🎯 Estrategia balanceada:")
print(f"   - Fine-tuning progresivo (80→50→0 capas congeladas)")
print(f"   - Dropout: 0.5 + 0.4 + 0.3")
print(f"   - Arquitectura: 512 → 256 neuronas")
print(f"   - Learning rate: 0.0001 con decaimiento")
print(f"   - Data augmentation agresivo")

model.summary()

# ---------------------------
# CALLBACKS AVANZADOS
# ---------------------------
# Early stopping estricto para overfitting < 5%
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,  # Muy paciente para permitir convergencia
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,  # Más paciencia antes de reducir LR
    min_lr=1e-7,
    verbose=1
)

checkpoint_path = MODELS_DIR / "best_mobilenetv2_balanced.keras"
model_checkpoint = ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Callback para control estricto de overfitting
class StrictOverfittingControl(tf.keras.callbacks.Callback):
    def __init__(self, max_overfitting=5.0):
        super().__init__()
        self.max_overfitting = max_overfitting
        self.best_gap = 100.0
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        if logs and 'accuracy' in logs and 'val_accuracy' in logs:
            current_gap = (logs['accuracy'] - logs['val_accuracy']) * 100
            
            if current_gap < self.best_gap:
                self.best_gap = current_gap
                self.best_weights = self.model.get_weights()
                
            if current_gap > self.max_overfitting:
                print(f"🚨 OVERFITTING ALTO: {current_gap:.2f}% > {self.max_overfitting}%")
                print("   Reduciendo capacidad de aprendizaje...")
                # Reducir learning rate adicionalmente
                print("   (No se puede ajustar manualmente el learning rate porque usa un scheduler)")

    
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            print(f"🎯 Restaurando mejores pesos (overfitting: {self.best_gap:.2f}%)")
            self.model.set_weights(self.best_weights)

callbacks = [
    ProgressiveUnfreezing(),
    early_stopping, 
    reduce_lr, 
    model_checkpoint, 
    StrictOverfittingControl(max_overfitting=5.0)
]

# ---------------------------
# ENTRENAMIENTO
# ---------------------------
print("🚀 Iniciando entrenamiento balanceado...")
print("OBJETIVO: >70% accuracy + Overfitting < 5%")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
print("Entrenamiento finalizado.")

# ---------------------------
# EVALUACIÓN COMPLETA
# ---------------------------
print("Calculando métricas finales...")
y_true = []
y_pred = []
y_pred_proba = []

for images, labels in val_dataset:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_proba.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)

# Encontrar la mejor época
best_epoch = np.argmax(history.history['val_accuracy'])
train_acc = history.history['accuracy'][best_epoch]
val_acc = history.history['val_accuracy'][best_epoch]
overfitting_gap = abs(train_acc - val_acc) * 100

print("\n" + "="*60)
print("📊 RESULTADOS FINALES - BALANCE PRECISIÓN/OVERFITTING")
print("="*60)
print(f"   🎯 Accuracy validación: {accuracy*100:.2f}%")
print(f"   📈 Mejor train accuracy: {train_acc*100:.2f}%")
print(f"   📉 Mejor validation accuracy: {val_acc*100:.2f}%")
print(f"   ⚖️  Gap de overfitting: {overfitting_gap:.2f}%")

# Evaluación de objetivos
if overfitting_gap < 5 and accuracy > 0.7:
    print("   🏆 ¡OBJETIVO CUMPLIDO! Alta precisión + bajo overfitting")
elif overfitting_gap < 5:
    print("   ✅ Overfitting controlado, precisión necesita mejora")
elif accuracy > 0.7:
    print("   ✅ Buena precisión, overfitting necesita control")
else:
    print("   ⚠️  Ambos objetivos necesitan mejora")

print("="*60)

# Guardar modelo final
final_model_path = MODELS_DIR / "mobilenetv2_balanced_final.keras"
model.save(final_model_path)
print(f"Modelo guardado en: {final_model_path}")

# ---------------------------
# SCRIPT DE PREDICCIÓN MEJORADO
# ---------------------------
def predict_garbage_material(image_path, confidence_threshold=70.0):
    """
    Predice el material de basura con control de confianza
    """
    try:
        # Cargar y preprocesar imagen
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Predecir
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        predicted_class_idx = np.argmax(score)
        predicted_class = class_names[predicted_class_idx]
        confidence = 100 * np.max(score)
        
        print(f"\n🔍 ANÁLISIS DE: {os.path.basename(image_path)}")
        print(f"   📦 Material identificado: {predicted_class}")
        print(f"   🎯 Nivel de confianza: {confidence:.2f}%")
        
        # Validación de confianza
        if confidence >= confidence_threshold:
            print(f"   ✅ PREDICCIÓN CONFIABLE (>{confidence_threshold}%)")
            return predicted_class, confidence
        else:
            print(f"   ⚠️  PREDICCIÓN POCO CONFIABLE (<{confidence_threshold}%)")
            print(f"   💡 Considere tomar otra foto con mejor iluminación/ángulo")
            
            # Mostrar alternativas
            top_3 = np.argsort(predictions[0])[-3:][::-1]
            print(f"   📊 Alternativas:")
            for i, idx in enumerate(top_3):
                conf = 100 * predictions[0][idx]
                print(f"      {i+1}. {class_names[idx]}: {conf:.2f}%")
            
            return predicted_class, confidence
            
    except Exception as e:
        print(f"❌ Error al procesar imagen: {e}")
        return None, 0.0

# Probar con una imagen de ejemplo si existe
example_dir = DATA_DIR / "test_example"
if example_dir.exists() and any(example_dir.iterdir()):
    example_image = next(example_dir.iterdir())
    if example_image.is_file():
        predicted, confidence = predict_garbage_material(str(example_image))
        print(f"\n🎯 Ejemplo de predicción: {predicted} ({confidence:.1f}%)")

# Guardar función de predicción
prediction_code = f"""
# PREDICTOR DE MATERIALES DE BASURA
# Confianza mínima: 70%

def predecir_material_basura(ruta_imagen):
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.models import load_model
    
    # Cargar modelo y clases
    model = load_model('{final_model_path}')
    class_names = {class_names}
    IMG_SIZE = (224, 224)
    
    # Preprocesar
    img = tf.keras.utils.load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Predecir
    predictions = model.predict(img_array, verbose=0)
    confidence = 100 * np.max(tf.nn.softmax(predictions[0]))
    material = class_names[np.argmax(predictions[0])]
    
    if confidence >= 70:
        return material, confidence
    else:
        return f"{{material}} (poca confianza: {{confidence:.1f}}%)", confidence

# Uso:
# material, confianza = predecir_material_basura('ruta/a/tu/imagen.jpg')
"""

with open(REPORTS_DIR / "predictor_materiales.py", "w", encoding="utf-8") as f:
    f.write(prediction_code)

print(f"📄 Predictor guardado en: {REPORTS_DIR / 'predictor_materiales.py'}")
