"""File locations (Colab defaults; set MOLMAGE_COLAB_HOME to use another root)."""
import os
import sys
from pathlib import Path

HOME = Path(os.environ.get("MOLMAGE_COLAB_HOME", "/content")).expanduser().absolute()

# visible in the Colab file browser
MOLMAGE_ROOT = HOME / "MolMage"            # official MolMage source: scripts/ utils/ src/ datasets/
OUT_ROOT = HOME / "molmage_outputs"        # one folder per molecule
LAST_RESULT = OUT_ROOT / "last_result.json"

# tooling, kept out of the way
TOOLS = HOME / ".molmage"
ENV_DIR = TOOLS / "env"                    # Python 3.10 + bpy 4.0.0 + PyMOL + Open3D + RDKit
ENV_PY = ENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
UV_DIR = TOOLS / "uv"
WHEEL_DIR = TOOLS / "wheels"
INPUT_DIR = TOOLS / "inputs"               # uploaded files
READY_MARKER = ENV_DIR / ".molmage_ready"

RUNNER = Path(__file__).resolve().with_name("runner.py")  # executed by ENV_PY


def clean_env(**extra):
    """Environment for the Python 3.10 subprocesses.

    Colab's PYTHONPATH, inline matplotlib backend, LD_PRELOAD and UV_* / PIP_* settings all
    target Colab's own Python, so they are dropped.
    """
    drop = ("PYTHONPATH", "PYTHONHOME", "MPLBACKEND", "LD_PRELOAD")
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith(("UV_", "PIP_"))}
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.update({k: str(v) for k, v in extra.items()})
    return env
