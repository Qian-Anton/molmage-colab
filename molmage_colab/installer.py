"""Create the Python 3.10 environment MolMage needs and fetch the official MolMage source."""
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

from . import paths
from .ui import Status

ANON_API = "https://anonymous.4open.science/api/repo/ImagenMol-65A8"
DOCKER_REPO = "panzq/molmage"
DOCKER_LAYER = "sha256:b10524055e2853a0d76a23d546eaa33473ba44963e9da31e76d36cc36b430deb"  # /example_molmage
UV_VERSION = "0.9.5"

# bpy 4.0.0 is only on Blender's own index (PyPI dropped it) and MolMage needs Blender < 4.1.
BPY_INDEX = "https://download.blender.org/pypi/bpy/"
BPY_WHEELS = {
    "linux": ("bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl",
              "d6caceb7598232df42dcaa13433c6fee91ea811c67a3e32816643840f97c30cf"),
    "win32": ("bpy-4.0.0-cp310-cp310-win_amd64.whl",
              "57e66dcf1a99e0bdbb8fbaad52e6c922b9454e3b9a06ce698852ab66bc6a31b1"),
}
OPEN3D = {"linux": "open3d-cpu==0.19.0", "win32": "open3d==0.19.0"}
PACKAGES = [  # pinned; pymol-open-source-whl requires numpy==1.26.4
    "numpy==1.26.4", "scipy==1.15.3", "scikit-image==0.25.2", "matplotlib==3.10.3",
    "pillow==11.3.0", "h5py==3.13.0", "pyyaml==6.0.2", "tqdm==4.67.1", "six==1.17.0",
    "einops==0.8.1", "trimesh==4.6.8", "PyMCubes==0.1.6", "libigl==2.5.0",
    "rdkit==2024.9.6", "pymol-open-source-whl==3.2.0.1",
]
APT_PACKAGES = ["libgl1", "libegl1", "libglu1-mesa", "libgomp1", "libx11-6", "libxrender1", "libxi6",
                "libxxf86vm1", "libxfixes3", "libxkbcommon0", "libsm6", "libice6", "libudev1"]
REQUIRED_FILES = ["scripts/encode_molmage.py", "scripts/decode_molmage.py", "utils/molutils.py",
                  "src/xgutils/omgutil.py", "src/xgutils/bpyutil.py", "src/xgutils/vis/visutil.py",
                  "src/xgutils/assets/preset_glb.blend"]
HTTP_HEADERS = {"User-Agent": "molmage-colab/0.1"}
PYATLAS_STUB = '''"""Placeholder: pyatlas (a UVAtlas binding) only ships inside the MolMage Docker image."""


def atlas(*args, **kwargs):
    raise RuntimeError("pyatlas is not available here; use the default Open3D UV method")
'''
SELF_TEST = """
import json, sys
sys.path[:0] = ["src", "."]
import numpy, rdkit, open3d, trimesh, igl, bpy
from pymol import cmd
from utils import molutils
from xgutils import omgutil
print(json.dumps({"Python": sys.version.split()[0], "bpy": bpy.app.version_string, "PyMOL": cmd.get_version()[0],
                  "Open3D": open3d.__version__, "RDKit": rdkit.__version__, "NumPy": numpy.__version__}))
"""


# --------------------------------------------------------------------------- helpers
def _sh(cmd, **kwargs):
    proc = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kwargs)
    if proc.returncode != 0:
        shown = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
        tail = (proc.stdout + proc.stderr).strip()[-3000:]
        raise RuntimeError(f"command failed (exit {proc.returncode}): {shown}\n{tail}")
    return proc.stdout


def _platform():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "win32"
    raise RuntimeError(f"Unsupported platform {sys.platform!r}: bpy 4.0.0 wheels are used for Linux (Colab) and Windows")


def _http_get(url, auth=None, timeout=60, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HTTP_HEADERS)
            if auth:  # not forwarded on redirects (the registry redirects blobs to a CDN)
                req.add_unredirected_header("Authorization", auth)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, dest, sha256, progress=None, retries=3):
    """Download ``url`` to ``dest`` and check its SHA256 (an existing valid file is reused)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HTTP_HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                total, done, shown = int(resp.headers.get("Content-Length") or 0), 0, -1
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress and total and int(20 * done / total) != shown:
                        shown = int(20 * done / total)
                        progress(done / 1e6, total / 1e6)
            if _sha256(dest) == sha256:
                return dest
            last = RuntimeError("SHA256 mismatch")
        except Exception as exc:
            last = exc
        time.sleep(3 * (attempt + 1))
    dest.unlink(missing_ok=True)
    raise RuntimeError(f"download failed: {url}: {last}")


def _find_uv(root):
    if not root.exists():
        return None
    for name in ("uv", "uv.exe"):
        for path in sorted(root.rglob(name)):
            if path.is_file() and path.parent.name in ("bin", "Scripts"):
                return path
    return None


def _uv():
    """A pinned uv in .molmage/uv (falls back to the uv that Colab ships)."""
    exe = _find_uv(paths.UV_DIR)
    if exe is None:
        try:
            _sh([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps",
                 "--upgrade", "--target", str(paths.UV_DIR), f"uv=={UV_VERSION}"], env=paths.clean_env())
        except RuntimeError:
            pass
        exe = _find_uv(paths.UV_DIR) or shutil.which("uv")
    if exe is None:
        raise RuntimeError("could not install uv")
    if Path(exe).suffix != ".exe":
        Path(exe).chmod(0o755)
    return [str(exe), "--no-config"]


def _uv_env():
    return paths.clean_env(UV_PYTHON_PREFERENCE="only-managed", UV_NO_PROGRESS="1", UV_HTTP_TIMEOUT="300",
                           UV_CACHE_DIR=paths.TOOLS / "uv-cache", UV_PYTHON_INSTALL_DIR=paths.TOOLS / "python")


# --------------------------------------------------------------------------- MolMage source
def _fetch_from_anonymous(tmp):
    """Mirror MolMage/ from the anonymous repository API, file by file (sizes checked)."""
    files = {}

    def walk(path):
        entries = json.loads(_http_get(f"{ANON_API}/files/?path={urllib.parse.quote(path)}"))
        if not isinstance(entries, list):
            raise RuntimeError(f"unexpected listing for {path}: {str(entries)[:200]}")
        for entry in entries:
            sub = f"{entry['path']}/{entry['name']}" if entry.get("path") else entry["name"]
            if "__pycache__" in sub:
                continue
            if "size" in entry:
                files[sub] = int(entry["size"])
            else:
                walk(sub)

    walk("MolMage")
    for sub, size in files.items():
        data = _http_get(f"{ANON_API}/file/{urllib.parse.quote(sub)}")
        if len(data) != size:
            raise RuntimeError(f"{sub}: got {len(data)} bytes, expected {size}")
        out = tmp / Path(sub).relative_to("MolMage")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return len(files)


def _fetch_from_docker_layer(tmp):
    """Extract /example_molmage from one layer of panzq/molmage (content-addressed, SHA256-checked)."""
    scope = f"repository:{DOCKER_REPO}:pull"
    token = json.loads(_http_get(f"https://auth.docker.io/token?service=registry.docker.io&scope={scope}"))["token"]
    blob = _http_get(f"https://registry-1.docker.io/v2/{DOCKER_REPO}/blobs/{DOCKER_LAYER}",
                     auth=f"Bearer {token}", timeout=120)
    if "sha256:" + hashlib.sha256(blob).hexdigest() != DOCKER_LAYER:
        raise RuntimeError("Docker layer SHA256 mismatch")
    count = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if (not member.isfile() or len(parts) < 2 or parts[0] != "example_molmage"
                    or ".." in parts or "__pycache__" in parts):
                continue
            out = tmp.joinpath(*parts[1:])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tar.extractfile(member).read())
            count += 1
    return count


def _validate_molmage(root):
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).is_file()]
    if missing:
        raise RuntimeError(f"missing files: {missing}")
    if (root / "src/xgutils/assets/preset_glb.blend").read_bytes()[:7] != b"BLENDER":
        raise RuntimeError("preset_glb.blend is not a valid .blend file")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for py in root.rglob("*.py"):
            compile(py.read_text(encoding="utf-8"), str(py), "exec")


def fetch_molmage(source="auto"):
    """Download the official MolMage code into paths.MOLMAGE_ROOT; returns the source used."""
    fetchers = {"anonymous-repo": _fetch_from_anonymous, "docker-layer": _fetch_from_docker_layer}
    order = list(fetchers) if source == "auto" else [source]
    errors = []
    for label in order:
        tmp = paths.TOOLS / "MolMage_download"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            fetchers[label](tmp)
            _validate_molmage(tmp)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue
        shutil.rmtree(paths.MOLMAGE_ROOT, ignore_errors=True)
        tmp.rename(paths.MOLMAGE_ROOT)
        return label
    raise RuntimeError("could not download the MolMage source:\n" + "\n".join(errors))


# --------------------------------------------------------------------------- setup
def _summary(versions):
    return " · ".join(f"{k} {v}" for k, v in versions.items())


def is_ready():
    return paths.READY_MARKER.exists() and (paths.MOLMAGE_ROOT / "utils" / "molutils.py").exists()


def setup(force=False, source="auto"):
    """Install everything MolMage needs (≈3–5 min the first time, instant afterwards).

    Builds a separate Python 3.10 environment (bpy 4.0.0 only exists for 3.10) and downloads the
    MolMage source from the anonymous repository, falling back to the panzq/molmage Docker image.
    """
    t0 = time.time()
    status = Status("Setting up MolMage ...")
    if force:
        shutil.rmtree(paths.ENV_DIR, ignore_errors=True)
        shutil.rmtree(paths.MOLMAGE_ROOT, ignore_errors=True)
    if is_ready():
        versions = json.loads(paths.READY_MARKER.read_text())
        status.update("Ready (already installed) · " + _summary(versions))
        return versions
    for folder in (paths.TOOLS, paths.OUT_ROOT, paths.INPUT_DIR, paths.WHEEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    plat = _platform()
    current = ["setup"]

    def step(k, text):
        current[0] = f"[{k}/6] {text}"
        status.update(f"Setting up MolMage {current[0]} ...")

    try:
        if not paths.READY_MARKER.exists():
            step(1, "system libraries")
            if plat == "linux" and shutil.which("apt-get"):
                install = ("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
                           + " ".join(APT_PACKAGES))
                try:
                    _sh(install)
                except RuntimeError:  # stale package lists: refresh once and retry
                    try:
                        _sh("apt-get update -qq")
                        _sh(install)
                    except RuntimeError:
                        pass  # most of these are already in Colab; the self-test reports anything missing

            step(2, "Python 3.10")
            uv = _uv()
            if not paths.ENV_PY.exists():
                _sh([*uv, "venv", "--python", "3.10", str(paths.ENV_DIR)], env=_uv_env())

            step(3, "bpy 4.0.0 wheel")
            name, sha = BPY_WHEELS[plat]
            wheel = download_verified(
                BPY_INDEX + name, paths.WHEEL_DIR / name, sha,
                progress=lambda done, total: status.update(
                    f"Setting up MolMage [3/6] bpy 4.0.0 wheel {done:.0f}/{total:.0f} MB ..."))

            step(4, "RDKit, PyMOL, Open3D, bpy and friends")
            _sh([*uv, "pip", "install", "--python", str(paths.ENV_PY), *PACKAGES, OPEN3D[plat], str(wheel)],
                env=_uv_env())

        step(5, "MolMage source")
        if not (paths.MOLMAGE_ROOT / "utils" / "molutils.py").exists():
            fetch_molmage(source)

        step(6, "self-test")
        site = _sh([str(paths.ENV_PY), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).strip()
        Path(site, "pyatlas.py").write_text(PYATLAS_STUB)
        out = _sh([str(paths.ENV_PY), "-c", SELF_TEST], cwd=str(paths.MOLMAGE_ROOT), env=paths.clean_env())
        found = [line for line in out.strip().splitlines() if line.startswith("{")]  # PyMOL / Blender also print
        if not found:
            raise RuntimeError(f"the self-test printed no result:\n{out[-2000:]}")
        versions = json.loads(found[-1])
    except Exception:
        status.update(f"Setup failed at {current[0]} (details below)")
        raise
    paths.READY_MARKER.write_text(json.dumps(versions))
    status.update(f"Ready in {time.time() - t0:.0f} s · " + _summary(versions))
    return versions
