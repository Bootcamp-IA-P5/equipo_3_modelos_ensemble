```markdown
# Docker guide (CPU) — EcoSort

Este documento explica cómo construir y ejecutar la imagen Docker optimizada para CPU.

Requisitos:
- Docker Desktop o Docker Engine instalado.
- Para GPU se necesitaría una máquina con GPU NVIDIA y drivers apropiados 

Construir la imagen (CPU):
- Desde la raíz del repo:
  docker build -f Dockerfile -t equipo_3_modelos_ensemble:cpu .

Levantar con docker-compose:
  docker compose up -d --build

Entrar al contenedor:
  docker exec -it e3m_app_cpu bash

Ejecutar scripts dentro del contenedor:
  python scripts/map_labels.py
  python scripts/predict_example.py --image tests/data/sample.jpg
  pytest -q tests/smoke_test.py

Ejecutar streamlit desde host sin compose:
  docker run --rm -p 8501:8501 -v "%cd%:/app" -w /app equipo_3_modelos_ensemble:cpu \
    streamlit run app/app_streamlit.py --server.port=8501 --server.address=0.0.0.0

Notas:
- NO incluyas modelos o datasets pesados en la imagen; móntalos como volúmenes.
- Si necesitas GPU en el futuro, usa Dockerfile.gpu y una máquina con GPU NVIDIA + drivers CUDA-on-WSL (o nube).
```