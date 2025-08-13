# executor.py
import subprocess
import tempfile
import os

def run_python_code(code: str):
    """
    Executes Python code in isolation and returns (stdout, stderr).
    This is dangerous if used with untrusted code — run in a sandbox for production.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
        f.write(code)
        f.flush()
        script_path = f.name

    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True
        )
        return result.stdout.strip(), result.stderr.strip()
    finally:
        os.remove(script_path)
