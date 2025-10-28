```text
# Tests - instrucciones

Para que CI echa a andar el smoke test automáticamente, añade un archivo de imagen de prueba:
- tests/data/sample.jpg (pequeña imagen JPG/PNG, < 200 KB)

Cómo ejecutar localmente:
1) Activar virtualenv y deps:
   python -m venv .venv
   .\.venv\Scripts\Activate    (Windows) o source .venv/bin/activate (Unix)
   pip install -r requirements.txt
   pip install pytest

2) Ejecutar test:
   pytest -q tests/smoke_test.py
```