import streamlit as st
from PIL import Image
import subprocess
import json
import os
import tempfile

# --- Configuración de la Página ---
st.set_page_config(
    page_title="EcoSort",
    page_icon="♻️",
    layout="centered",
)

# --- Estilos CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; color: #2E7D32; text-align: center; font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 12px;
        padding: 10px 24px; border: none; font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- Título y Descripción ---
st.markdown('<p class="main-header">EcoSort: Tu Asistente de Reciclaje ♻️</p>', unsafe_allow_html=True)
st.write("Sube una foto de un residuo y nuestra IA te dirá cómo reciclarlo.")

# --- Carga de la Imagen ---
uploaded_file = st.file_uploader(
    "Elige una imagen...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada.', use_column_width=True)

    if st.button('Clasificar Residuo'):
        with st.spinner('Analizando la imagen... 🧐'):
            # Guardar la imagen subida en un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                image.save(tmpfile.name)
                tmp_image_path = tmpfile.name

            try:
                # --- Llamada al Script de Predicción ---
                # Usamos subprocess para ejecutar el script de predicción como un
                # proceso separado. Esto define el "contrato" entre el frontend y el backend.
                command = ["python", "scripts/predict.py", "--image", tmp_image_path]

                # Ejecutar el comando y capturar la salida
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                # La salida del script es una cadena JSON, la parseamos
                response = json.loads(result.stdout)

                prediction = response.get('prediction', 'No se pudo determinar')
                confidence = response.get('confidence', 0)

                st.success('¡Clasificación completada!')
                st.write(f"## **Resultado:** {prediction.replace('_', ' ').title()}")
                st.write(f"**Confianza:** {confidence:.2%}")

            except subprocess.CalledProcessError as e:
                st.error("Error al ejecutar el modelo de clasificación.")
                st.code(f"Detalles del error:\n{e.stderr}")
            except json.JSONDecodeError:
                st.error("Error al procesar la respuesta del modelo. Respuesta no válida.")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
            finally:
                # Eliminar el archivo temporal
                os.remove(tmp_image_path)

