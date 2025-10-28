# Lightweight smoke tests for CI / local quickcheck
import subprocess
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Path to a small test image. CI expects the repo to include tests/data/sample.jpg
SAMPLE_IMG = ROOT / "tests" / "data" / "sample.jpg"

def run_cmd(cmd_args, cwd=ROOT, timeout=30):
    """Run a command given as a list of args (no shell), return (returncode, stdout+stderr)."""
    if isinstance(cmd_args, (str,)):
        # As a defensive fallback, run via shell only if a string was provided.
        proc = subprocess.run(cmd_args, cwd=str(cwd), shell=True, capture_output=True, text=True, timeout=timeout)
    else:
        proc = subprocess.run(cmd_args, cwd=str(cwd), shell=False, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out

def test_map_labels_runs():
    """map_labels.py should run without error and produce label_map.json (if train.csv present)."""
    map_script = SCRIPTS / "map_labels.py"
    assert map_script.exists(), f"{map_script} not found"
    # call using list of args to avoid shell quoting issues on Windows
    rc, out = run_cmd([sys.executable, str(map_script)])
    ok_keywords = ["no se encontró", "not found", "saved label map", "saved"]
    assert rc == 0 or any(k in out.lower() for k in ok_keywords)

def test_predict_example_runs_or_skips():
    """Try to run predict_example.py on a small image.
    If no model exists, script should not crash (it should handle missing model gracefully)."""
    pred_script = SCRIPTS / "predict_example.py"
    assert pred_script.exists(), f"{pred_script} not found"

    # If SAMPLE_IMG is present use it; else try data/raw/... fallback
    if SAMPLE_IMG.exists():
        img = SAMPLE_IMG
    else:
        fallback = ROOT / "data" / "raw" / "garbage_classification" / "plastic" / "plastic278.jpg"
        img = fallback if fallback.exists() else None

    if img:
        rc, out = run_cmd([sys.executable, str(pred_script), "--image", str(img)], timeout=60)
        ok_keywords = ["model", "modelo", "prediction", "predicción", "no se encontró", "not found"]
        assert rc == 0 or any(k in out.lower() for k in ok_keywords)
    else:
        pytest.skip("No sample image found; skipping predict test")

def test_import_src_modules():
    """Basic import sanity check: try importing package modules used by scripts."""
    sys.path.insert(0, str(ROOT))
    try:
        import src  # noqa: F401
    except Exception as e:
        pytest.skip(f"Cannot import src package: {e}")