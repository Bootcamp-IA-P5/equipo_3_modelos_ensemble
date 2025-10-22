# 1. Usar una imagen base ligera de Python
FROM python:3.10-slim

# 2. Configurar el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de requisitos e instalar dependencias
# (Aprovechamos la caché de Docker al hacer esto primero)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código de la aplicación
# Asumimos que la carpeta 'app' contiene app_streamlit.py
# y la carpeta 'scripts' contiene predict.py
COPY . /app

# 5. Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# 6. Comando para ejecutar la aplicación Streamlit
# Streamlit debe ejecutarse en el puerto 8501 y en el host 0.0.0.0
# para ser accesible fuera del contenedor.
CMD ["streamlit", "run", "app/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]