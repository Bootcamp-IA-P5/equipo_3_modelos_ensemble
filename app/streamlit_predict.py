import streamlit as st
from pathlib import Path
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# Reducir logs de TF
tf.get_logger().setLevel("ERROR")

# Intentar importar funciones de models
try:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
    from ensemble import load_models, build_generators
except Exception:
    try:
        from models.ensemble import load_models, build_generators
    except Exception:
        load_models = None
        build_generators = None

# Rutas por defecto a modelos (ajusta si cambias nombres)
DEFAULT_MODEL_PATHS = [
    "models/trained/efficientnetb3_20251023-102542.h5",
    "models/trained/mobilenetv2_20251022-144833.h5",
    "models/trained/resnet50_20251022-141919.h5",
    "models/trained/vgg16_20251022-144302.h5",
]

# Mapeo orientativo de clases a contenedores en España (varía por municipio)
BIN_MAP = {
    "cardboard": ("Azul", "Papel y cartón (contenedor azul)"),
    "paper": ("Azul", "Papel y cartón (contenedor azul)"),
    "glass": ("Verde", "Vidrio (iglú verde)"),
    "metal": ("Amarillo", "Envases (metálico) - contenedor amarillo"),
    "plastic": ("Amarillo", "Envases plásticos - contenedor amarillo"),
    "trash": ("Gris", "Resto / Orgánico según municipio - fracción resto (gris)"),
}

st.set_page_config(page_title="Clasificador de basura - Ensamble", layout="wide")

st.markdown(
    "<h1 style='text-align:center'>Clasificador de basura (Ensamble)</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center'>Sube una imagen con drag & drop o usa el selector. La predicción se hace con un ensamble de modelos entrenados.</p>",
    unsafe_allow_html=True,
)

# Mensaje de estado global de carga de modelos
if st.session_state.get("models_loaded"):
    st.success(f"Modelos cargados: {len(st.session_state.get('models', []))}")
elif st.session_state.get("model_load_error"):
    st.error("Error al cargar modelos automáticamente. Revisa el log del servidor.")


def _discover_model_paths():
    """Descubre modelos .h5 en models/trained. Si no hay, intenta DEFAULT_MODEL_PATHS existentes."""
    repo_root = Path(__file__).resolve().parents[1]
    trained_dir = repo_root / "models" / "trained"
    paths = []
    if trained_dir.exists():
        h5s = sorted(trained_dir.glob("*.h5"))
        paths.extend([str(p.resolve()) for p in h5s])
    if not paths:
        for p in DEFAULT_MODEL_PATHS:
            pp = Path(p)
            if not pp.is_absolute():
                pp = (repo_root / p).resolve()
            if pp.exists():
                paths.append(str(pp))
    return paths


# Sidebar (sin mostrar rutas de modelos)
with st.sidebar:
    st.header("Configuración")
    use_generator_for_classes = st.checkbox(
        "Obtener nombres de clases desde generador (data/processed)", value=True
    )
    st.markdown("---")
    st.markdown("Los modelos se cargan automáticamente al iniciar la app.")


@st.cache_resource
def load_models_and_fns(model_paths):
    repo_root = Path(__file__).resolve().parents[1]

    if load_models is None:
        raise RuntimeError("No se puede importar load_models desde models.ensemble")

    resolved_candidates = []
    resolved_for_load = []
    diagnostics = []

    # Resolver cada entrada intentando varias ubicaciones plausibles
    for p in model_paths:
        p_in = (p or "").strip()
        if p_in == "":
            continue

        tried = []
        found = None
        p_path = Path(p_in)

        # Candidate: as given (absolute or relative to cwd)
        tried.append(str(p_path))
        if p_path.exists():
            found = p_path.resolve()

        # Candidate: relative to repo root
        if found is None:
            cand = (repo_root / p_in).resolve()
            tried.append(str(cand))
            if cand.exists():
                found = cand

        # Candidate: inside models/ or models/trained folders
        if found is None:
            cand = (repo_root / "models" / p_in).resolve()
            tried.append(str(cand))
            if cand.exists():
                found = cand

        if found is None:
            cand = (repo_root / "models" / "trained" / Path(p_in).name).resolve()
            tried.append(str(cand))
            if cand.exists():
                found = cand

        # Candidate: search by filename under repo (first match)
        if found is None:
            name = Path(p_in).name
            matches = list(repo_root.glob(f"**/{name}"))
            tried.extend([str(m) for m in matches[:5]])
            if matches:
                found = matches[0].resolve()

        diagnostics.append(
            {
                "input": p_in,
                "resolved": str(found) if found is not None else None,
                "exists": bool(found),
                "tried": tried,
            }
        )
        if found is not None:
            resolved_for_load.append(str(found))

    # Cargar modelos sólo desde las rutas resueltas que existen
    models = load_models(resolved_for_load)

    # construir predict functions trazadas para cada modelo (mitiga retracing)
    predict_fns = []
    for name, m in models:
        # crear función trazada ligada al modelo actual
        @tf.function
        def _fn(x, _m=m):
            return _m(x, training=False)

        predict_fns.append((name, _fn))

    return models, predict_fns, diagnostics


def get_class_names_from_generator():
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if build_generators is None:
            return None
        val_gen = build_generators(
            str(repo_root / "data" / "processed" / "garbage_classification_split")
        )
        class_names = [
            k for k, v in sorted(val_gen.class_indices.items(), key=lambda x: x[1])
        ]
        return class_names
    except Exception:
        return None


def preprocess_image(pil_img, img_size=(224, 224)):
    img = pil_img.convert("RGB").resize(img_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_from_fns(arr, predict_fns):
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    preds = []
    if not predict_fns:
        raise ValueError(
            "No hay funciones de predicción disponibles. Carga los modelos primero."
        )
    for name, fn in predict_fns:
        p = fn(tensor).numpy()
        if p.ndim == 2 and p.shape[0] == 1:
            p = p[0]
        else:
            p = np.ravel(p)
        preds.append(p)
    all_preds = np.array(preds)
    avg = np.mean(all_preds, axis=0)
    return avg


# Carga de modelos en el arranque (sin exponer rutas en UI)
if "models_loaded" not in st.session_state:
    try:
        with st.spinner("Cargando modelos..."):
            auto_paths = _discover_model_paths()
            models, predict_fns, diagnostics = load_models_and_fns(auto_paths)
        st.session_state["models"] = models
        st.session_state["predict_fns"] = predict_fns
        st.session_state["model_load_diagnostics"] = diagnostics  # guardado interno
        st.session_state["models_loaded"] = bool(models)
    except Exception as e:
        st.session_state["models_loaded"] = False
        st.session_state["model_load_error"] = str(e)

# Interfaz principal
uploaded = st.file_uploader(
    "Arrastra aquí una imagen o pulsa para seleccionar",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

col_main, col_side = st.columns([3, 1])

with col_side:
    # Mostrar estado sin revelar rutas
    if st.session_state.get("models_loaded"):
        st.success(f"Modelos cargados: {len(st.session_state.get('models', []))}")
        st.write("Los modelos están listos para predecir.")
    else:
        err = st.session_state.get("model_load_error")
        if err:
            st.error("No se pudieron cargar los modelos automáticamente.")
        else:
            st.info("Cargando modelos...")


# Intentar obtener class names
class_names = None
if use_generator_for_classes:
    class_names = get_class_names_from_generator()
if class_names is None:
    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

with col_main:
    if uploaded is None:
        st.info("Sube una imagen para ver la predicción (drag & drop).")
    else:
        img = Image.open(uploaded)

        # Mostrar imagen y resultados lado a lado (responsive)
        left, right = st.columns([1, 1])
        with left:
            # Use width='stretch' en versiones recientes de Streamlit
            try:
                st.image(img, caption="Imagen subida", width="stretch")
            except TypeError:
                # Fallback para versiones anteriores
                st.image(img, caption="Imagen subida", use_container_width=True)

        with right:
            if st.session_state.get("models_loaded", False):
                if st.button("Predecir"):
                    with st.spinner("Calculando predicción..."):
                        arr = preprocess_image(img)
                        avg = predict_from_fns(arr, st.session_state["predict_fns"])

                        # Asegurar que avg sea array
                        if np.ndim(avg) == 0:
                            avg = np.array([avg])

                        # Comprobar compatibilidad entre class_names y avg
                        if len(class_names) != len(avg):
                            st.error(
                                f"Número de clases ({len(class_names)}) y tamaño de salida del modelo ({len(avg)}) no coinciden. Mostraré el resultado por índice."
                            )
                            probs = {
                                (
                                    class_names[i]
                                    if i < len(class_names)
                                    else f"idx_{i}"
                                ): float(avg[i])
                                for i in range(len(avg))
                            }
                            idx = int(np.argmax(avg))
                            name = (
                                class_names[idx]
                                if idx < len(class_names)
                                else f"idx_{idx}"
                            )
                        else:
                            idx = int(np.argmax(avg))
                            name = class_names[idx]
                            probs = {cn: float(p) for cn, p in zip(class_names, avg)}

                    st.subheader(f"Predicción: {name} (idx={idx})")
                    st.write("Probabilidades:")
                    st.table(
                        {"clase": list(probs.keys()), "prob": list(probs.values())}
                    )

                    # Sugerencia de contenedor según BIN_MAP
                    bin_info = BIN_MAP.get(name, ("Desconocido", "No hay sugerencia"))
                    st.info(f"Contenedor sugerido: {bin_info[0]} — {bin_info[1]}")
                    st.caption(
                        "Nota: esta recomendación es orientativa y puede variar según el municipio. Consulta la normativa local para casos concretos."
                    )

                    # Barra de probabilidades
                    try:
                        import pandas as pd

                        df = pd.DataFrame(
                            {"prob": list(probs.values())}, index=list(probs.keys())
                        )
                        st.bar_chart(df)
                    except Exception:
                        pass
            else:
                st.warning(
                    "Los modelos no están cargados. Vuelve a cargar la página o revisa el log del servidor."
                )

st.markdown("---")
st.markdown("### Notas sobre la clasificación y normativa (España)")
st.markdown("- Papel y cartón → contenedor azul.")
st.markdown("- Envases plásticos y metálicos → contenedor amarillo (envases ligeros).")
st.markdown("- Vidrio → iglú verde.")
st.markdown("- Resto → contenedor gris / fracción resto (varía por municipio).")
st.markdown(
    "Estas son orientaciones generales; consulta la normativa local de tu ayuntamiento para validación definitiva."
)
