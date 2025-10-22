# Dockerfile para EcoSort - Clasificador de Residuos

FROM python:3.10-slim

# Metadatos
LABEL maintainer="Equipo 3 - Bootcamp F5 IA"
LABEL description="Aplicación de clasificación de residuos con Deep Learning y Ensemble"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/feedback data/feedback/images models/trained

# Exponer puerto de Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando por defecto
CMD ["streamlit", "run", "app/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
