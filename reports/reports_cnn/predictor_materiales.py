
# PREDICTOR DE MATERIALES DE BASURA
# Confianza mínima: 70%

def predecir_material_basura(ruta_imagen):
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.models import load_model
    
    # Cargar modelo y clases
    model = load_model('..\reports\models\mobilenetv2_balanced_final.keras')
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
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
        return f"{material} (poca confianza: {confidence:.1f}%)", confidence

# Uso:
# material, confianza = predecir_material_basura('ruta/a/tu/imagen.jpg')
