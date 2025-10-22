import streamlit as st
from PIL import Image
import subprocess
import json
import os
import tempfile
import base64

# --- Configuración de la Página ---
st.set_page_config(
    page_title="EcoSort",
    page_icon="♻️",
    layout="centered",
)

# --- 1. Definición del Historial y Colores ---
# Inicializar el historial en el estado de sesión si no existe
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Mapeo de colores y contenedores (Para resultados dinámicos)
CLASSIFICATION_MAP = {
    "cardboard": {"icon": "📦", "info": "Contenedor Azul. Asegúrate de que esté limpio y seco.", "color": "#0099ff"}, # Azul
    "glass": {"icon": "🍾", "info": "Contenedor Verde. Enjuaga los envases.", "color": "#28a745"},   # Verde
    "metal": {"icon": "🥫", "info": "Contenedor Amarillo. Límpialas antes de desecharlas.", "color": "#ffc107"},   # Amarillo
    "paper": {"icon": "📄", "info": "Contenedor Azul. Evita el papel manchado de grasa.", "color": "#007bff"},    # Azul
    "plastic": {"icon": "🧴", "info": "Contenedor Amarillo. Revisa los números de reciclaje.", "color": "#ffc107"}, # Amarillo
    "trash": {"icon": "🗑️", "info": "Contenedor Gris/Restos. No es reciclable.", "color": "#6c757d"}   # Gris
}

# --- 2. Funciones Auxiliares ---

# Función para convertir la imagen a Base64 para guardarla en el historial
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- Estilos CSS Mejorados ---
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.8rem; color: #1e7e34; text-align: center; font-weight: bold;
        text-shadow: 2px 2px 4px #aaa;
    }}
    .stButton>button {{
        background-color: #28a745; color: white; border-radius: 12px;
        padding: 10px 24px; border: none; font-size: 18px; font-weight: bold;
        transition: background-color 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #1e7e34;
    }}
    
    /* Estilo para el resultado dinámico */
    .result-box {{
        padding: 20px;
        border-radius: 15px;
        margin-top: 15px;
        border: 2px solid; 
        font-size: 1.2rem;
    }}
    .confidence-text {{
        font-size: 1.1rem;
        font-weight: bold;
    }}
    
    /* Estilo para el historial */
    .history-item {{
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f8f9fa;
        display: flex;
        gap: 15px;
        align-items: center;
    }}
</style>
""", unsafe_allow_html=True)


# --- Título y Descripción ---
st.markdown('<p class="main-header">EcoSort: Tu Asistente de Reciclaje 🌱♻️</p>', unsafe_allow_html=True)
st.write("Sube una foto de un residuo y nuestra IA te dirá cómo reciclarlo, ¡ayuda a nuestro planeta!")

# --- Contenedor Principal (Tabs) ---
tab1, tab2 = st.tabs(["Clasificación en Vivo", "Historial de Reciclaje"])

with tab1:
    # --- Carga de la Imagen ---
    uploaded_file = st.file_uploader(
        "Elige una imagen de un residuo...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen cargada.', use_container_width=True)

        if st.button('Clasificar Residuo'):
            with st.spinner('Analizando la imagen... 🧐'):
                # Guardar la imagen subida en un archivo temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    image.save(tmpfile.name)
                    tmp_image_path = tmpfile.name
                    
                # Obtener la imagen en Base64 para el historial (antes de eliminarla)
                image_b64 = get_image_base64(tmp_image_path)

                try:
                    # --- Llamada al Script de Predicción ---
                    command = ["python", "scripts/predict.py", "--image", tmp_image_path]

                    # Ejecutar el comando y capturar la salida
                    result = subprocess.run(command, capture_output=True, text=True, check=True)

                    # La salida del script es una cadena JSON, la parseamos
                    response = json.loads(result.stdout)

                    prediction = response.get('prediction', 'No se pudo determinar')
                    confidence = response.get('confidence', 0)
                    
                    st.success('¡Clasificación completada!')
                    
                    # --- Bloque de Visualización de Resultados y Registro ---
                    key = prediction.lower().replace(' ', '_')
                    info_data = CLASSIFICATION_MAP.get(key, {"icon": "❓", "info": "No se encontró información.", "color": "#6c757d"})
                    
                    # 3. Guardar en el Historial
                    st.session_state['history'].insert(0, { # insertamos al inicio (más reciente)
                        'prediction': prediction.replace('_', ' ').title(),
                        'confidence': confidence,
                        'info': info_data['info'],
                        'color': info_data['color'],
                        'icon': info_data['icon'],
                        'image_b64': image_b64
                    })

                    # Mostrar el resultado con estilo dinámico
                    st.markdown(f"""
                    <div class="result-box" style="border-color: {info_data['color']}; color: #333;">
                        <h2>{info_data['icon']} Resultado: {prediction.replace('_', ' ').title()}</h2>
                        <p class="confidence-text">Confianza: {confidence:.2%}</p>
                        <p><strong>Guía de Reciclaje:</strong> {info_data['info']}</p>
                    </div>
                    """, unsafe_allow_html=True)

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

# --- 4. Sección de Historial ---
with tab2:
    st.header("Historial de Clasificaciones")
    if st.session_state['history']:
        
        if st.button("Limpiar Historial"):
            st.session_state['history'] = []
            st.rerun()

        for item in st.session_state['history']:
            # Muestra cada elemento del historial en un formato atractivo
            st.markdown(f"""
            <div class="history-item">
                <img src="data:image/jpeg;base64,{item['image_b64']}" style="width: 80px; height: 80px; object-fit: cover; border-radius: 8px;">
                <div>
                    <h4>{item['icon']} {item['prediction']}</h4>
                    <p style="color: {item['color']}; margin: 0;">{item['info']}</p>
                    <p style="font-size: 0.9rem; margin: 0;">Confianza: {item['confidence']:.2%}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aún no tienes clasificaciones. ¡Sube una imagen en la pestaña anterior para empezar!")