import pytest
import os
import subprocess

@pytest.mark.smoke
def test_predict_script_runs():
    """
    Smoke test to ensure the predict.py script runs without errors.
    It checks for the script's existence and runs it with --help.
    """
    script_path = "scripts/predict.py"
    
    if not os.path.exists(script_path):
        pytest.skip(f"'{script_path}' not found, skipping smoke test.")
        
    try:
        result = subprocess.run(
            ["python", script_path, "--help"],
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError if return code is non-zero
            encoding='utf-8'
        )
        
        # Check for a successful exit code (already done by check=True)
        assert result.returncode == 0
        
        # Check for a key part of the help message
        assert "usage: predict.py" in result.stdout
        
    except FileNotFoundError:
        pytest.fail(f"Python interpreter not found. Cannot run '{script_path}'.")
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Running '{script_path} --help' failed with exit code {e.returncode}.\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}"
        )
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")
