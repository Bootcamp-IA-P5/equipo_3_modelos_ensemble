#!/usr/bin/env python3
"""
Divide requirements.txt en:
 - requirements-prod.txt  (dependencias mínimas para producción)
 - requirements-dev.txt   (dependencias de desarrollo / plataforma específicas)

Uso:
  python run_split_requirements.py --input requirements.txt --prod requirements-prod.txt --dev requirements-dev.txt

La lista de patrones DEV_FILTERS contiene paquetes que NO deben ir al build de producción
(p. ej. pywinpty, jupyter, ipykernel, pytest, black, flake8, torch (lo movemos a dev por si quieres
instalar una versión CPU-only manualmente), nvidia/cuda/triton, etc.).
"""
import argparse
import re
from pathlib import Path

# Patrones (substrings, case-insensitive) que consideramos "dev / platform-specific / heavy"
DEV_FILTERS = [
    "pywinpty",
    "ipykernel",
    "jupyter",
    "jupyterlab",
    "notebook",
    "ipython",
    "pytest",
    "black",
    "flake8",
    "mypy",
    "torch",
    "torchvision",
    "triton",
    "nvidia",
    "cuda",
    "pywinpty-bindings",
    "pywinpty",
    "pyqt",
    "maturin",
    # agrega más patrones si sabes de otros paquetes problemáticos
]

def is_dev_line(line: str):
    L = line.lower()
    if not L or L.startswith("#"):
        return False
    for pat in DEV_FILTERS:
        if pat in L:
            return True
    return False

def normalize_line(line: str) -> str:
    return line.rstrip("\n")

def split_requirements(input_path: Path, prod_out: Path, dev_out: Path):
    prod_lines = []
    dev_lines = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        line = normalize_line(raw)
        if not line:
            continue
        # If line is a direct URL (git+https://...) keep in prod unless it matches dev filters
        if is_dev_line(line):
            dev_lines.append(line)
        else:
            prod_lines.append(line)

    # write outputs
    prod_out.write_text("\n".join(prod_lines) + ("\n" if prod_lines and not prod_lines[-1].endswith("\n") else ""), encoding="utf-8")
    dev_out.write_text("\n".join(dev_lines) + ("\n" if dev_lines and not dev_lines[-1].endswith("\n") else ""), encoding="utf-8")

    print(f"Wrote {len(prod_lines)} lines to {prod_out}")
    print(f"Wrote {len(dev_lines)} lines to {dev_out}")
    if dev_lines:
        print("Moved the following dev/filtered packages to", dev_out, ":")
        for l in dev_lines:
            print("  -", l)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="requirements.txt")
    parser.add_argument("--prod", "-p", default="requirements-prod.txt")
    parser.add_argument("--dev", "-d", default="requirements-dev.txt")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input requirements file not found: {input_path}")
    split_requirements(input_path, Path(args.prod), Path(args.dev))

if __name__ == "__main__":
    main()
    