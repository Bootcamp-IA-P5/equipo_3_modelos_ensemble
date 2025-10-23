"""
Aplicación Streamlit para clasificación de residuos con sistema de feedback.
"""

import streamlit as st
from PIL import Image
import subprocess
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qrcode
from io import BytesIO

# Generar QR dinámicamente
qr_url = "https://ecosort.streamlit.app"  # Cambia por tu URL real
qr_img = qrcode.make(qr_url)

# Convertir a BytesIO para mostrar sin guardar en disco
buffer = BytesIO()
qr_img.save(buffer, format="PNG")
buffer.seek(0)

# Mostrar en Streamlit
st.image(buffer, caption="📱 Escanea para acceder desde tu móvil", width=250)


# --- Configuración de la Página ---
st.set_page_config(
    page_title="EcoSort - Clasificador de Residuos",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #f0f8f0;
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Inicializar estado de la sesión ---
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# --- Funciones auxiliares ---
FEEDBACK_DIR = Path('data/feedback')
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

def save_feedback(image_path, prediction, confidence, correct_label, user_comment):
    """Guarda el feedback del usuario."""
    feedback = {
        'timestamp': datetime.now().isoformat(),
        'image_path': image_path,
        'prediction': prediction,
        'confidence': confidence,
        'correct_label': correct_label,
        'user_comment': user_comment,
        'correct': prediction == correct_label
    }
    
    # Guardar en archivo JSON
    feedback_file = FEEDBACK_DIR / f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(feedback_file, 'w') as f:
        json.dump(feedback, f, indent=2)
    
    # Copiar imagen a directorio de feedback
    if os.path.exists(image_path):
        feedback_img_dir = FEEDBACK_DIR / 'images'
        feedback_img_dir.mkdir(exist_ok=True)
        
        img_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        feedback_img_path = feedback_img_dir / img_name
        
        img = Image.open(image_path)
        img.save(feedback_img_path)
    
    st.session_state.feedback_data.append(feedback)
    
    return True

def load_feedback_stats():
    """Carga estadísticas de feedback."""
    feedback_files = list(FEEDBACK_DIR.glob('feedback_*.json'))
    
    if not feedback_files:
        return None
    
    feedbacks = []
    for file in feedback_files:
        with open(file, 'r') as f:
            feedbacks.append(json.load(f))
    
    return pd.DataFrame(feedbacks)

def get_available_experiments():
    """Obtiene lista de experimentos disponibles."""
    trained_dir = Path("models/trained")
    if not trained_dir.exists():
        return []
    experiments = sorted([d for d in trained_dir.glob("exp_*") if d.is_dir()], reverse=True)
    return experiments

def get_available_models(experiment_dir):
    """Obtiene los modelos ensemble disponibles en un experimento."""
    models = {
        'Stacking Ensemble': 'stacking_model.pkl',
        'Voting Ensemble': 'voting_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'LightGBM': 'lightgbm_model.pkl'
    }
    
    available = {}
    for name, filename in models.items():
        model_path = experiment_dir / filename
        if model_path.exists():
            available[name] = str(model_path)
    
    return available

def classify_image(image_path, ensemble_model_path=None):
    """Clasifica una imagen usando el script de predicción."""
    try:
        # Buscar el último experimento si no se especifica modelo
        experiments = get_available_experiments()
        
        if not experiments:
            st.error("❌ No se encontraron modelos entrenados. Por favor, entrena un modelo primero.")
            st.info("Ejecuta: `python scripts/train_models.py --epochs 30 --batch_size 32 --img_size 128 --model_name resnet18`")
            return None
        
        latest_exp = experiments[0]
        cnn_model = str(latest_exp / "best_model.pth")
        
        # Verificar que el modelo CNN existe
        if not Path(cnn_model).exists():
            st.error(f"❌ Modelo CNN no encontrado en: {cnn_model}")
            return None
        
        # Usar el modelo ensemble especificado o el por defecto
        if ensemble_model_path is None:
            ensemble_model = str(latest_exp / "stacking_model.pkl")
        else:
            ensemble_model = ensemble_model_path
        
        # Verificar que el modelo ensemble existe
        if not Path(ensemble_model).exists():
            st.error(f"❌ Modelo ensemble no encontrado en: {ensemble_model}")
            return None
        
        # Obtener el directorio raíz del proyecto
        import sys
        project_root = Path(__file__).parent.parent
        
        # Buscar el Python del entorno virtual
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            # Intentar con linux/mac path
            venv_python = project_root / ".venv" / "bin" / "python"
        
        # Si no existe el venv, usar sys.executable como fallback
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        command = [
            python_executable,
            str(project_root / "scripts" / "predict.py"), 
            "--image", image_path,
            "--cnn_model", str(project_root / cnn_model),
            "--ensemble_model", str(project_root / ensemble_model)
        ]
        
        # Ejecutar con encoding UTF-8 explícito para Windows
        result = subprocess.run(
            command, 
            capture_output=True, 
            encoding='utf-8',
            errors='ignore',  # Ignorar caracteres que no se puedan decodificar
            check=True, 
            timeout=30, 
            cwd=str(project_root)
        )
        
        # Debug: Verificar qué se recibió
        stdout_clean = result.stdout.strip() if result.stdout else ""
        
        if stdout_clean:
            try:
                response = json.loads(stdout_clean)
                return response
            except json.JSONDecodeError as je:
                st.error("❌ Error al decodificar JSON de la respuesta")
                with st.expander("🔍 Ver detalles del problema de JSON"):
                    st.code(f"stdout recibido:\n{repr(stdout_clean)}\n\nError: {str(je)}")
                return None
        
        st.error("❌ El modelo no devolvió ninguna respuesta")
        with st.expander("🔍 Ver información de diagnóstico"):
            st.code(f"stdout vacío: {len(result.stdout) if result.stdout else 0} caracteres\n\nstderr:\n{result.stderr}")
        return None
    
    except subprocess.CalledProcessError as e:
        st.error("❌ Error al ejecutar el modelo de clasificación.")
        with st.expander("🔍 Ver detalles del error"):
            st.code(f"Comando ejecutado:\n{' '.join(command)}\n\nError (stderr):\n{e.stderr}\n\nSalida (stdout):\n{e.stdout}")
        st.info("💡 Verifica que todos los paquetes estén instalados: `pip install -r requirements.txt`")
        st.warning(f"🐍 Python usado: {sys.executable}")
        return None
    except json.JSONDecodeError as e:
        st.error("❌ Error al procesar la respuesta del modelo.")
        with st.expander("🔍 Ver detalles del error"):
            st.code(f"Respuesta recibida (stdout):\n{repr(result.stdout)}\n\nError de JSON: {str(e)}\n\nLogs (stderr):\n{result.stderr}")
        return None
    except subprocess.TimeoutExpired:
        st.error("❌ La predicción tardó demasiado tiempo (timeout de 30 segundos)")
        st.info("💡 Esto puede ocurrir si el modelo es muy grande o la imagen muy compleja")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {type(e).__name__}: {str(e)}")
        with st.expander("🔍 Ver stack trace completo"):
            import traceback
            st.code(traceback.format_exc())
        st.warning(f"🐍 Python usado: {sys.executable}")
        st.info("💡 Asegúrate de estar usando el entorno virtual correcto")
        return None

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/recycle-sign.png", width=80)
    st.title("🗑️ EcoSort")
    st.markdown("---")
    
    page = st.radio(
        "Navegación",
        ["🏠 Clasificar", "📊 Métricas", "📝 Feedback", "ℹ️ Acerca de"],
        key="navigation_radio"
    )
    
    st.markdown("---")
    st.markdown("### Clases soportadas:")
    classes_info = {
        '📦': 'Cardboard (Cartón)',
        '🍾': 'Glass (Vidrio)',
        '🔩': 'Metal',
        '📄': 'Paper (Papel)',
        '🥤': 'Plastic (Plástico)',
        '🗑️': 'Trash (Basura)'
    }
    
    for emoji, name in classes_info.items():
        st.markdown(f"{emoji} {name}")

# --- Página principal: Clasificar ---
if page == "🏠 Clasificar":
    st.markdown('<p class="main-header">♻️ EcoSort: Clasificador Inteligente de Residuos</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sube una foto de un residuo y nuestra IA te dirá cómo reciclarlo correctamente</p>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Subir Imagen")
        
        # Selector de experimento y modelo
        experiments = get_available_experiments()
        
        if experiments:
            st.markdown("### 🧪 Configuración del Modelo")
            
            # Selector de experimento
            exp_names = [f"{exp.name} ({datetime.fromtimestamp(exp.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})" 
                        for exp in experiments]
            selected_exp_idx = st.selectbox(
                "Selecciona experimento:",
                range(len(exp_names)),
                format_func=lambda i: exp_names[i],
                help="Selecciona qué experimento de entrenamiento usar"
            )
            
            selected_exp = experiments[selected_exp_idx]
            
            # Selector de modelo ensemble
            available_models = get_available_models(selected_exp)
            
            if available_models:
                selected_model_name = st.selectbox(
                    "Selecciona modelo ensemble:",
                    list(available_models.keys()),
                    help="Prueba diferentes modelos para comparar resultados"
                )
                selected_model_path = available_models[selected_model_name]
                
                # Mostrar info del experimento
                results_file = selected_exp / "final_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        st.info(f"📊 Accuracy del experimento: {results.get('test_accuracy', 'N/A')}")
            else:
                st.warning("⚠️ No se encontraron modelos ensemble en este experimento")
                selected_model_path = None
        else:
            st.error("❌ No hay modelos entrenados disponibles")
            st.info("Ejecuta primero: `python scripts/train_models.py --epochs 30 --batch_size 32 --img_size 128`")
            selected_model_path = None
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Elige una imagen...",
            type=["jpg", "jpeg", "png"],
            help="Sube una foto clara del residuo que quieres clasificar"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen cargada', width='stretch')
            
            if st.button('🔍 Clasificar Residuo', disabled=(selected_model_path is None)):
                with st.spinner('Analizando la imagen con IA... 🧐'):
                    # Guardar imagen temporal
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                        image.save(tmpfile.name)
                        tmp_image_path = tmpfile.name
                    
                    # Clasificar con el modelo seleccionado
                    response = classify_image(tmp_image_path, selected_model_path)
                    
                    if response and 'prediction' in response:
                        st.session_state.last_prediction = response
                        st.session_state.last_image_path = tmp_image_path
                        st.session_state.used_model = selected_model_name
                        st.rerun()
    
    with col2:
        if 'last_prediction' in st.session_state:
            result = st.session_state.last_prediction
            
            st.markdown("### 📊 Resultados")
            
            # Mostrar modelo usado
            if 'used_model' in st.session_state:
                st.info(f"🤖 Modelo usado: **{st.session_state.used_model}**")
            
            prediction = result.get('prediction', 'unknown')
            confidence = result.get('confidence', 0)
            recycling_info = result.get('recycling_info', {})
            probabilities = result.get('probabilities', {})
            
            # Resultado principal
            st.success('✅ ¡Clasificación completada!')
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #000000; margin-bottom: 10px;">
                    {recycling_info.get('name', prediction.title())}
                </h2>
                <h3 style="color: #000000;">{recycling_info.get('bin', '')}</h3>
                <p style="font-size: 1.1rem; margin-top: 15px; color: #000000;">
                    <strong>Confianza:</strong> {confidence*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Barra de confianza
            st.progress(confidence)
            
            # Tips de reciclaje
            st.markdown("### 💡 Consejos de Reciclaje")
            st.info(recycling_info.get('tips', 'No hay consejos disponibles'))
            
            # Distribución de probabilidades
            st.markdown("### 📈 Distribución de Probabilidades")
            
            prob_df = pd.DataFrame({
                'Clase': list(probabilities.keys()),
                'Probabilidad': [v*100 for v in probabilities.values()]
            })
            prob_df = prob_df.sort_values('Probabilidad', ascending=False)
            
            # Crear gráfico con matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(prob_df['Clase'], prob_df['Probabilidad'], color='#2E7D32', alpha=0.7)
            ax.set_xlabel('Clase', fontsize=12, fontweight='bold')
            ax.set_ylabel('Probabilidad (%)', fontsize=12, fontweight='bold')
            ax.set_title('Confianza por Clase', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Sistema de feedback
            st.markdown("---")
            st.markdown("### 📝 ¿Es correcta esta clasificación?")
            
            col_fb1, col_fb2 = st.columns(2)
            
            with col_fb1:
                if st.button("✅ Correcto", key="btn_correct"):
                    save_feedback(
                        st.session_state.last_image_path,
                        prediction,
                        confidence,
                        prediction,
                        "Clasificación correcta"
                    )
                    st.success("¡Gracias por tu feedback!")
            
            with col_fb2:
                if st.button("❌ Incorrecto", key="btn_incorrect"):
                    st.session_state.show_feedback_form = True
            
            if st.session_state.get('show_feedback_form', False):
                st.markdown("#### Ayúdanos a mejorar")
                
                correct_label = st.selectbox(
                    "¿Cuál es la clase correcta?",
                    ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
                )
                
                user_comment = st.text_area(
                    "Comentarios adicionales (opcional)",
                    placeholder="¿Por qué crees que falló la clasificación?"
                )
                
                if st.button("📮 Enviar Feedback"):
                    save_feedback(
                        st.session_state.last_image_path,
                        prediction,
                        confidence,
                        correct_label,
                        user_comment
                    )
                    st.success("¡Feedback guardado! Gracias por ayudarnos a mejorar 🎉")
                    st.session_state.show_feedback_form = False
                    st.rerun()

# --- Página de Métricas ---
elif page == "📊 Métricas":
    st.markdown('<p class="main-header">📊 Métricas en Tiempo Real</p>', unsafe_allow_html=True)
    
    feedback_df = load_feedback_stats()
    
    if feedback_df is not None and len(feedback_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predicciones", len(feedback_df))
        
        with col2:
            correct_preds = feedback_df['correct'].sum()
            accuracy = (correct_preds / len(feedback_df)) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        with col3:
            avg_confidence = feedback_df['confidence'].mean()
            st.metric("Confianza Promedio", f"{avg_confidence*100:.1f}%")
        
        with col4:
            st.metric("Feedback Recibido", len(feedback_df))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Distribución de Predicciones")
            pred_counts = feedback_df['prediction'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%',
                   startangle=90)
            ax.set_title('Predicciones por Clase', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### ✅ Accuracy por Clase")
            accuracy_by_class = feedback_df.groupby('prediction')['correct'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(accuracy_by_class.index, accuracy_by_class.values, color='#4CAF50', alpha=0.7)
            ax.set_xlabel('Clase', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Accuracy por Clase', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("### 📈 Evolución Temporal")
        feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
        daily_accuracy = feedback_df.groupby('date')['correct'].mean() * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_accuracy.index, daily_accuracy.values, marker='o', 
                linewidth=2, markersize=8, color='#2E7D32')
        ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy a lo Largo del Tiempo', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("📭 No hay datos de feedback todavía. ¡Empieza a clasificar y proporcionar feedback!")

# --- Página de Feedback ---
elif page == "📝 Feedback":
    st.markdown('<p class="main-header">📝 Historial de Feedback</p>', unsafe_allow_html=True)
    
    feedback_df = load_feedback_stats()
    
    if feedback_df is not None and len(feedback_df) > 0:
        st.markdown(f"### Total de registros: {len(feedback_df)}")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_class = st.multiselect(
                "Filtrar por clase",
                options=feedback_df['prediction'].unique()
            )
        
        with col2:
            filter_correct = st.selectbox(
                "Filtrar por correctitud",
                ["Todos", "Correctos", "Incorrectos"]
            )
        
        # Aplicar filtros
        filtered_df = feedback_df.copy()
        
        if filter_class:
            filtered_df = filtered_df[filtered_df['prediction'].isin(filter_class)]
        
        if filter_correct == "Correctos":
            filtered_df = filtered_df[filtered_df['correct'] == True]
        elif filter_correct == "Incorrectos":
            filtered_df = filtered_df[filtered_df['correct'] == False]
        
        # Mostrar tabla
        st.dataframe(
            filtered_df[['timestamp', 'prediction', 'confidence', 'correct_label', 'correct']]
        )
        
        # Exportar datos
        if st.button("📥 Exportar datos para reentrenamiento"):
            export_path = FEEDBACK_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filtered_df.to_csv(export_path, index=False)
            st.success(f"✅ Datos exportados a: {export_path}")
    
    else:
        st.info("📭 No hay feedback registrado todavía.")

# --- Página Acerca de ---
elif page == "ℹ️ Acerca de":
    st.markdown('<p class="main-header">ℹ️ Acerca de EcoSort</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🌍 Misión
    
    EcoSort es una aplicación de inteligencia artificial diseñada para facilitar el reciclaje
    correcto de residuos. Nuestro objetivo es ayudar a las personas a clasificar sus residuos
    de manera más eficiente y contribuir a un planeta más sostenible.
    
    ### 🤖 Tecnología
    
    - **Deep Learning**: Modelos de redes neuronales convolucionales (CNN) pre-entrenadas
    - **Transfer Learning**: ResNet18 optimizado para 6GB de VRAM
    - **Ensemble Learning**: Combinación de Random Forest, XGBoost y LightGBM
    - **Embeddings**: Extracción de características de alto nivel
    
    ### 📊 Rendimiento
    
    - 6 clases de residuos
    - Optimizado para hardware limitado
    - Sistema de feedback en tiempo real
    - Mejora continua mediante reentrenamiento
    
    ### 👥 Equipo
    
    Proyecto desarrollado por el **Equipo 3** del Bootcamp F5 IA
    
    ### 📞 Contacto
    
    ¿Tienes sugerencias o encontraste algún problema? 
    ¡Tu feedback es muy valioso para nosotros!
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Recursos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("🔗 [GitHub](https://github.com/Factoria-F5-madrid/equipo_3_modelos_ensemble)")
    
    with col2:
        st.markdown("📖 [Documentación](#)")
    
    with col3:
        st.markdown("🎓 [Bootcamp F5](#)")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>♻️ EcoSort - Clasificador Inteligente de Residuos | Equipo 3 - Bootcamp F5 IA</p>
        <p style="font-size: 0.9rem;">Desarrollado con ❤️ para un planeta más sostenible 🌍</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --- NOTA: Sección de Historial comentada (falta definir tabs) ---
# Si quieres habilitar historial, agrega st.tabs() en la página de Clasificar
# y descomenta el siguiente código:

# # --- 4. Sección de Historial ---
# with tab2:
#     st.header("Historial de Clasificaciones")
#     if st.session_state['history']:
#         
#         if st.button("Limpiar Historial"):
#             st.session_state['history'] = []
#             st.rerun()
#
#         for item in st.session_state['history']:
#             # Muestra cada elemento del historial en un formato atractivo
#             st.markdown(f"""
#             <div class="history-item">
#                 <img src="data:image/jpeg;base64,{item['image_b64']}" style="width: 80px; height: 80px; object-fit: cover; border-radius: 8px;">
#                 <div>
#                     <h4>{item['icon']} {item['prediction']}</h4>
#                     <p style="color: {item['color']}; margin: 0;">{item['info']}</p>
#                     <p style="font-size: 0.9rem; margin: 0;">Confianza: {item['confidence']:.2%}</p>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#     else:
#         st.info("Aún no tienes clasificaciones. ¡Sube una imagen en la pestaña anterior para empezar!")