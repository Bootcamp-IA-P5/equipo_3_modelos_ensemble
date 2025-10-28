FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Copiar requirements y script de división
COPY requirements.txt run_split_requirements.py ./

# Instalar dependencias incluyendo torch CPU
RUN python -m pip install --upgrade pip setuptools wheel && \
    python run_split_requirements.py --input requirements.txt --prod requirements-prod.txt --dev requirements-dev.txt && \
    # Instalar torch CPU explícitamente
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0+cpu torchvision==0.24.0+cpu && \
    # Instalar otras dependencias
    pip install --no-cache-dir -r requirements-prod.txt

# Copiar el código de la aplicación
COPY . .

# Verificar que los modelos existen
RUN ls -l models/trained/exp_20251022_110227/ || echo "Warning: No models found in expected path"

EXPOSE 8501

# Comando con configuraciones explícitas
CMD ["streamlit", "run", "app/app_streamlit.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.serverAddress", "localhost"]