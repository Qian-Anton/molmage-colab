"""Run the MolMage runner in the Python 3.10 environment and keep track of the results."""
import json
import re
import subprocess
from pathlib import Path

from . import paths
from .ui import Status, note

EXAMPLES = ("aspirin.sdf", "caffeine.sdf", "ibuprofen.sdf", "lidocaine.sdf", "acetaminophen.sdf",
            "ibuprofen.sdf.gz")
_STEP = re.compile(r"^\[molmage\] > (.+?) \.\.\.$")


class Result:
    """One processed molecule / MolMage; mirrors ``result.json`` in its output folder."""

    def __init__(self, info, log=""):
        self.info = info
        self.log = log

    @property
    def ok(self):
        return self.info.get("status") == "ok"

    @property
    def name(self):
        return self.info.get("mol_id", "")

    @property
    def out_dir(self):
        return Path(self.info["out_dir"]) if self.info.get("out_dir") else None

    @property
    def error(self):
        return self.info.get("error", "")

    def file(self, suffix):
        """Path of ``<out_dir>/<name><suffix>``, e.g. ``result.file("_mmg.glb")``."""
        return self.out_dir / f"{self.name}{suffix}"

    @property
    def molmage_file(self):
        found = sorted(self.out_dir.glob(f"{self.name}_molmage*.npy")) if self.out_dir else []
        return found[0] if found else None

    def molmage(self):
        """The MolMage tensor, shape (H, W, 4): x, y, z in [0, 1] and occupancy."""
        import numpy as np

        return np.load(self.molmage_file)

    def __repr__(self):
        if not self.ok:
            return f"Result(failed: {self.error})"
        shape = "x".join(map(str, self.info.get("molmage_shape", [])))
        return f"Result({self.name!r}, MolMage {shape}, {self.out_dir})"


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def latest():
    """The most recent result of :func:`run` (also after a kernel restart), or None."""
    info = _read(paths.LAST_RESULT)
    if info is None:
        return None
    if info.get("status") == "ok" and info.get("out_dir"):
        info = _read(Path(info["out_dir"]) / "result.json") or info  # picks up later renders
    return Result(info)


def _require(result):
    """``result`` or the latest one; prints a short note and returns None if there is nothing usable."""
    result = result if result is not None else latest()
    if result is None:
        note("Nothing to show yet: run step 2 first.")
        return None
    if not result.ok:
        note(f"Nothing to show: the last run failed ({result.error})")
        return None
    return result


def _installed():
    if paths.ENV_PY.exists() and (paths.MOLMAGE_ROOT / "utils" / "molutils.py").exists():
        return True
    note("The environment is not installed yet: run step 1 (mc.setup()) first.")
    return False


def _call(command, args, title, result_json, verbose=False):
    status = Status(f"{title} ...")
    cmd = [str(paths.ENV_PY), str(paths.RUNNER), command, "--molmage_root", str(paths.MOLMAGE_ROOT),
           "--output_dir", str(paths.OUT_ROOT), "--result_json", str(result_json), *map(str, args)]
    lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", cwd=str(paths.HOME), env=paths.clean_env())
    for line in proc.stdout:
        lines.append(line)
        if verbose:
            print(line, end="", flush=True)
        step = _STEP.match(line.strip())
        if step:
            status.update(f"{title} · {step.group(1)} ...")
    proc.wait()
    log = "".join(lines)
    (paths.OUT_ROOT / f"last_{command}.log").write_text(log, encoding="utf-8")
    info = _read(result_json)
    if info is None:
        status.update(f"{title} failed: the runner exited with code {proc.returncode}")
        print(log[-4000:])
        raise RuntimeError(f"runner exited with code {proc.returncode}")
    if info.get("out_dir") and Path(info["out_dir"]).is_dir():
        Path(info["out_dir"], "run.log" if command != "render" else "render.log").write_text(log, encoding="utf-8")
    if info.get("status") != "ok":
        status.update(f"{title} failed · {info.get('error')}")
        if proc.returncode != 2:  # unexpected error (not a user-input problem): show the log tail
            print("".join(lines[-30:]))
    return info, log, status


def _upload():
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError("Uploading only works in Colab; set FILE to a file path instead.") from None
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No file was uploaded.")
    name, data = next(iter(uploaded.items()))
    paths.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = paths.INPUT_DIR / Path(name).name
    path.write_bytes(data)
    return path


def _resolve_file(kind, file):
    file = (file or "").strip()
    if file.lower() in ("", "upload"):
        return _upload()
    if file in EXAMPLES:
        if kind != "sdf":
            raise ValueError(f"{file} is an SDF example: set INPUT to 'SDF file', or give a MolMage file.")
        return paths.MOLMAGE_ROOT / "datasets" / file
    path = Path(file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _pt_to_npy(path, index):
    """ImagenMol's generate.py saves MolMages as torch tensors of shape (1, 4, 64, 64)."""
    import numpy as np
    import torch  # preinstalled in Colab

    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        tensor = next(v for v in tensor.values() if torch.is_tensor(v))
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[index]
    out = paths.INPUT_DIR / f"{path.stem}_{index}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    return out


def _kind(text):
    text = (text or "").strip().lower()
    for kind in ("smiles", "sdf", "molmage"):
        if text.startswith(kind):
            return kind
    raise ValueError(f"INPUT must be 'SMILES', 'SDF file' or 'MolMage file', not {text!r}")


def run(input="SMILES", smiles="", file="upload", index=0, resolution=64, seed=42, skip_filters=False,
        name="", show=True, verbose=False):
    """Molecule -> MolMage -> mesh (or load an existing MolMage), then show one row of images.

    input        : "SMILES", "SDF file" or "MolMage file" (.npy / .pt)
    smiles       : the SMILES string (SMILES input)
    file         : "upload", an example name such as "aspirin.sdf", or a path
    index        : record in a multi-molecule SDF, or sample in a batched .pt
    resolution   : MolMage size (64 = official, 128)
    seed         : RDKit conformer seed for SMILES (-1 = the original unseeded embedding)
    skip_filters : run molecules that MolMage's input filters would reject
    """
    if not _installed():
        return None
    try:
        kind = _kind(input)
        if kind == "smiles":
            if not (smiles or "").strip():
                raise ValueError("SMILES is empty.")
            args = [f"--smiles={smiles.strip()}", "--seed", int(seed), "--resolution", int(resolution)]
            title, command = "Encoding", "encode"
        else:
            path = _resolve_file(kind, file)
            if kind == "sdf":
                args = [f"--sdf={path}", "--sdf_index", int(index), "--resolution", int(resolution)]
                title, command = f"Encoding {path.name}", "encode"
            else:
                if path.suffix == ".pt":
                    path = _pt_to_npy(path, int(index))
                args = [f"--npy={path}"]
                title, command = f"Decoding {path.name}", "load"
    except (ValueError, OSError, RuntimeError) as exc:
        note(f"Input problem: {exc}")
        return None
    if name and name.strip():
        args.append(f"--mol_id={name.strip()}")
    if skip_filters and command == "encode":
        args.append("--skip_filters")

    info, log, status = _call(command, args, title, paths.LAST_RESULT, verbose=verbose)
    result = Result(info, log)
    if result.ok:
        from .display import caption, show_molmage

        status.update(caption(result))
        if show:
            show_molmage(result, with_caption=False)
    return result


def render(result=None, views=4, size=512, samples=32, verbose=False):
    """Blender snapshots of the decoded mesh: the official preview plus ``views - 1`` turntable views."""
    result = _require(result)
    if result is None or not _installed():
        return None
    info, log, status = _call("render", ["--result_dir", result.out_dir, "--views", int(views), "--size", int(size),
                                         "--samples", int(samples)],
                              f"Rendering {result.name} ({int(views)} view{'s' if int(views) > 1 else ''})",
                              paths.OUT_ROOT / "last_render.json", verbose=verbose)
    if info.get("status") != "ok":
        return None
    seconds = info.get("render", {}).get("seconds")
    status.update(f"{result.name} · {int(views)} Blender snapshot{'s' if int(views) > 1 else ''} · {seconds} s")
    return Result(_read(result.out_dir / "result.json") or info, log)
