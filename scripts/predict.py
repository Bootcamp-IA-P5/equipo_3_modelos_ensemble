import argparse
import json
import random
import time
import sys

def predict(image_path: str):
    """
    Simulates a model prediction.
    In a real scenario, this function would load a trained model and
    perform inference on the image located at `image_path`.

    This placeholder version returns a random prediction to allow for
    parallel frontend development.
    """
    # Lista de clases posibles (debe coincidir con las del dataset)
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    # Simular el tiempo de procesamiento del modelo
    time.sleep(random.uniform(0.5, 2.0))

    # Generar una predicción y confianza aleatorias
    predicted_class = random.choice(classes)
    confidence = random.uniform(0.75, 0.99)

    # Devolver el resultado como un diccionario
    return {
        "prediction": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    # Configurar el parser para aceptar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Predict trash classification from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")

    args = parser.parse_args()

    # Validar que la ruta de la imagen no esté vacía (aunque no la usemos aquí)
    if not args.image:
        error_response = {"error": "Image path is required."}
        print(json.dumps(error_response))
        sys.exit(1)

    # Realizar la predicción "dummy"
    result = predict(args.image)

    # Imprimir el resultado como una cadena JSON a la salida estándar.
    # La aplicación de Streamlit leerá esta salida.
    print(json.dumps(result))
