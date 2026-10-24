#!/usr/bin/env python3
"""MolMage pipeline, executed inside the Python 3.10 environment created by molmage_colab.setup().

  encode : SMILES / SDF -> RDKit conformer -> PyMOL surface -> Open3D UV atlas
           -> Blender UV repack + bake (1024^2) -> MolMage (H, W, 4) -> mesh + GLB
  load   : an existing MolMage .npy -> mesh + GLB
  render : Blender snapshots of the decoded mesh (official preview + turntable)

The steps call the functions used by MolMage/scripts/encode_molmage.py and decode_molmage.py with
the same settings. Differences:
  * a failing step raises a readable error instead of being skipped silently;
  * the 20/30/30 s thread-pool timeouts are bypassed (``__wrapped__``), so slow CPUs do not abort
    large molecules and bpy stays on the main thread;
  * the UV-atlas preview is drawn once instead of once per triangle (same image, much faster);
  * SMILES conformers use a fixed RDKit seed by default (--seed -1 = the original unseeded smi2mol);
  * extra files for the notebook viewers (surface mesh + atoms, decoded mesh arrays, snapshots).

This file must stay standalone: it runs under a different interpreter than the rest of the package.
"""
import argparse
import contextlib
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

TAG = "[molmage]"
ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def log(msg):
    print(f"{TAG} {msg}", flush=True)


class StepError(RuntimeError):
    """A pipeline step failed; the message is shown to the notebook user."""


class Steps:
    """Times each pipeline step and tags failures with the step name."""

    def __init__(self):
        self.timings = {}
        self.current = None

    @contextlib.contextmanager
    def run(self, name):
        self.current = name
        log(f"> {name} ...")
        t0 = time.time()
        try:
            yield
        except StepError:
            raise
        except Exception as exc:  # report any failure together with the step name
            raise StepError(f"{name} failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self.timings[name] = round(time.time() - t0, 2)
        log(f"  done in {self.timings[name]:.1f} s")


def parse_args(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--molmage_root", required=True, help="MolMage folder (scripts/, utils/, src/)")
    common.add_argument("--output_dir", required=True, help="results go to <output_dir>/<mol_id>/")
    common.add_argument("--result_json", default=None, help="status file (default: <output_dir>/last_result.json)")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", parents=[common], help="SMILES / SDF -> MolMage -> mesh")
    src = enc.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="one SMILES string")
    src.add_argument("--sdf", help="a .sdf / .sdf.gz / .mol file")
    enc.add_argument("--sdf_index", type=int, default=0, help="0-based record index inside the SDF")
    enc.add_argument("--mol_id", default="", help="output name (default: SDF _Name / file name / SMILES hash)")
    enc.add_argument("--resolution", type=int, default=64, help="MolMage resolution, a divisor of 1024")
    enc.add_argument("--seed", type=int, default=42, help="RDKit seed for SMILES; -1 = original unseeded smi2mol")
    enc.add_argument("--o3d_size", type=int, default=1024, help="Open3D UV atlas size (official: 1024)")
    enc.add_argument("--o3d_gutter", type=int, default=10, help="Open3D UV atlas gutter (official: 10)")
    enc.add_argument("--skip_filters", action="store_true", help="ignore MolMage's check_smiles / check_sdf")
    enc.add_argument("--no_glb", action="store_true", help="skip the GLB / .blend export")

    load = sub.add_parser("load", parents=[common], help="existing MolMage .npy -> mesh")
    load.add_argument("--npy", required=True, help="MolMage .npy, shape (H, W, 4), (4, H, W) or (1, 4, H, W)")
    load.add_argument("--mol_id", default="", help="output name (default: file name)")
    load.add_argument("--no_glb", action="store_true", help="skip the GLB / .blend export")

    ren = sub.add_parser("render", parents=[common], help="Blender snapshots of a result folder")
    ren.add_argument("--result_dir", required=True, help="a folder written by 'encode' or 'load'")
    ren.add_argument("--views", type=int, default=4, help="number of turntable views")
    ren.add_argument("--size", type=int, default=512, help="snapshot size in px (official: 512)")
    ren.add_argument("--samples", type=int, default=32, help="Cycles samples (official: 32)")

    args = p.parse_args(argv)
    if args.command == "encode" and (args.resolution <= 0 or 1024 % args.resolution):
        p.error("--resolution must divide 1024, e.g. 32, 64 or 128")
    if args.command == "render" and not 1 <= args.views <= 36:
        p.error("--views must be between 1 and 36")
    return args


# --------------------------------------------------------------------------- helpers
def import_molmage(root):
    root = Path(root).resolve()
    if not (root / "utils" / "molutils.py").exists():
        raise StepError(f"MolMage source not found under {root}")
    for path in (root, root / "src"):  # same order as the official scripts (src first)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:  # utils/molutils.py imports pyatlas, a UVAtlas binding that only ships in the Docker image
        import pyatlas  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pyatlas")

        def _unavailable(*_args, **_kwargs):
            raise RuntimeError("pyatlas is not installed; use the default Open3D UV method")

        stub.atlas = _unavailable
        sys.modules["pyatlas"] = stub
    from utils import molutils
    from xgutils import omgutil
    from xgutils.vis import visutil

    return molutils, omgutil, visutil


def raw(fn):
    """MolMage wraps several steps in a thread-pool timeout; return the plain function."""
    return getattr(fn, "__wrapped__", fn)


def sanitize_id(text):
    text = re.sub(r"[^A-Za-z0-9_]+", "_", (text or "").strip()).strip("_")
    return text[:60]


def _size_reasons(mol):
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

    reasons = []
    heavy, rot = mol.GetNumHeavyAtoms(), CalcNumRotatableBonds(mol)
    if heavy > 50:
        reasons.append(f"{heavy} heavy atoms (limit 50)")
    if heavy < 6:
        reasons.append(f"{heavy} heavy atoms (minimum 6)")
    if rot > 10:
        reasons.append(f"{rot} rotatable bonds (limit 10)")
    extra = sorted({a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED_ELEMENTS)
    if extra:
        reasons.append("elements outside H C N O F P S Cl Br I: " + " ".join(extra))
    return reasons


def _char_reasons(smiles, label):
    bad = [c for c in "+-./\\" if c in smiles]
    if not bad:
        return []
    return [f"{label} contains {' '.join(bad)} (charges, explicit '-' bonds, "
            "salts / multiple fragments and E/Z marks are filtered out)"]


def explain_smiles(smiles):
    from rdkit import Chem

    reasons = _char_reasons(smiles, "the SMILES string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return reasons + ["RDKit cannot parse the SMILES"]
    return reasons + _size_reasons(mol)


def explain_sdf(mol):
    from rdkit import Chem

    if mol is None:
        return ["RDKit cannot parse this SDF record"]
    reasons = []
    conf = mol.GetConformer()
    flat = [a.GetIdx() + 1 for a in mol.GetAtoms() if conf.GetAtomPosition(a.GetIdx()).z == 0]
    if flat:
        reasons.append(f"atom(s) {flat[:5]} have z == 0, which MolMage treats as a 2D structure "
                       "(use SMILES input to get a 3D conformer)")
    reasons += _size_reasons(mol)
    reasons += _char_reasons(Chem.MolToSmiles(mol, isomericSmiles=False), "its canonical SMILES")
    return reasons


def read_sdf_blocks(path):
    path = Path(path)
    data = path.read_bytes()
    if path.name.lower().endswith(".gz"):
        data = gzip.decompress(data)
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return [block for block in text.split("$$$$\n") if block.strip()]


def embed_seeded(smiles, seed):
    """molutils.smi2mol with a fixed seed (AddHs -> ETKDG -> MMFF)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) != 0:
            raise StepError("RDKit could not embed a 3D conformer for this molecule")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def draw_2d(smiles, path):
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(400, 400)).save(path)
    except Exception as exc:  # cosmetic only
        log(f"  (2D depiction skipped: {exc})")


def draw_uv_atlas(mesh_uv, path, width=512, height=512):
    """Same drawing as molutils.mesh2uv(pngsavePath=...), saved once at the end."""
    from PIL import Image, ImageDraw

    uvs = np.asarray(mesh_uv.visual.uv)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for tri in np.asarray(mesh_uv.faces):
        draw.polygon([(int(uvs[k][0] * width), int(uvs[k][1] * height)) for k in tri], outline=(0, 0, 0))
    image.save(path)


def tile(frames, cols):
    h, w, c = frames[0].shape
    rows = -(-len(frames) // cols)
    canvas = np.zeros((rows * h, cols * w, c), dtype=np.float32)
    for i, frame in enumerate(frames):
        r, k = divmod(i, cols)
        canvas[r * h:(r + 1) * h, k * w:(k + 1) * w] = frame[..., :c]
    return canvas


def normalize_molmage(arr):
    """Accept (H, W, 4), (4, H, W) or (1, 4, H, W); same conventions as decode_molmage.py."""
    molmage = np.asarray(arr)
    if molmage.dtype == np.uint16:  # as omgutil.load_omg
        molmage = molmage.astype(np.float64) / 65535.0
    molmage = molmage.astype(np.float64)
    if molmage.ndim == 4 and molmage.shape[0] == 1:
        molmage = molmage[0]
    if molmage.ndim == 3 and molmage.shape[0] == 4 and molmage.shape[-1] != 4:
        molmage = np.transpose(molmage, (2, 1, 0)).copy()  # exactly what decode_molmage.py does
    if molmage.ndim != 3 or molmage.shape[-1] != 4:
        raise StepError(f"Invalid MolMage shape {tuple(np.shape(arr))}; expected (H, W, 4) or (4, H, W)")
    occ = molmage[..., 3] > 0.5
    molmage[~occ, :] = 0
    if not occ.any():
        raise StepError("The occupancy channel is empty; there is nothing to decode")
    return molmage


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("rdkit", "open3d", "trimesh"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import bpy

        out["bpy"] = bpy.app.version_string
    except Exception:
        pass
    try:
        from pymol import cmd

        out["pymol"] = cmd.get_version()[0]
    except Exception:
        pass
    return out


def fresh_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def namer(mol_dir, mol_id):
    def f(suffix):  # MolMage splits paths on "/", so always hand it POSIX paths
        return (mol_dir / f"{mol_id}{suffix}").as_posix()

    return f


# --------------------------------------------------------------------------- encode
def encode(args, mm, steps, info):
    molutils, omgutil, _visutil = mm
    from rdkit import Chem

    out_root = Path(args.output_dir).resolve()
    if args.smiles is not None:
        smiles_in = args.smiles.strip()
        if not smiles_in:
            raise StepError("The SMILES string is empty")
        with steps.run("check input"):
            canonical = molutils.check_smiles(smiles_in)
            if not canonical:
                parsed = Chem.MolFromSmiles(smiles_in)
                reasons = explain_smiles(smiles_in) or ["rejected by molutils.check_smiles"]
                if parsed is None or not args.skip_filters:
                    hint = " Tick SKIP_FILTERS to run it anyway." if parsed is not None else ""
                    raise StepError("MolMage's input filter rejected this SMILES: " + "; ".join(reasons) + "." + hint)
                canonical = Chem.MolToSmiles(parsed, isomericSmiles=False)
                info["filter_warnings"] = reasons
        mol_id = sanitize_id(args.mol_id) or "mol_" + hashlib.sha1(canonical.encode()).hexdigest()[:8]
        mol = None
        info.update(kind="smiles", input=smiles_in, seed=args.seed)
    else:
        with steps.run("read SDF"):
            blocks = read_sdf_blocks(args.sdf)
            if not blocks:
                raise StepError("No molecule record found in the SDF file")
            if not 0 <= args.sdf_index < len(blocks):
                raise StepError(f"INDEX={args.sdf_index} is out of range: the file has {len(blocks)} record(s)")
            mol = Chem.MolFromMolBlock(blocks[args.sdf_index])  # same call as molutils.read_sdf_from
            if molutils.check_sdf(mol) is None:
                reasons = explain_sdf(mol) or ["rejected by molutils.check_sdf"]
                if mol is None or not args.skip_filters:
                    hint = " Tick SKIP_FILTERS to run it anyway." if mol is not None else ""
                    raise StepError("MolMage's input filter rejected this SDF record: " + "; ".join(reasons) + "." + hint)
                info["filter_warnings"] = reasons
            canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            stem = Path(args.sdf).name
            for ext in (".sdf.gz", ".mol.gz", ".sdf", ".mol", ".gz"):
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            mol_id = sanitize_id(args.mol_id) or sanitize_id(name) or sanitize_id(stem) or "mol"
        info.update(kind="sdf", input=Path(args.sdf).name, sdf_index=args.sdf_index, sdf_records=len(blocks))

    mol_dir = fresh_dir(out_root / mol_id)
    info.update(mol_id=mol_id, smiles=canonical, out_dir=mol_dir.as_posix())
    f = namer(mol_dir, mol_id)
    Path(f(".txt")).write_text(canonical)
    draw_2d(canonical, f("_2d.png"))

    with steps.run("3D conformer (RDKit)"):
        if mol is None:
            mol = molutils.smi2mol(smi=canonical, sdfPath=f(".sdf")) if args.seed < 0 else embed_seeded(canonical, args.seed)
        with open(f(".sdf"), "w") as fh:
            fh.write(molutils.mol_to_sdfblock(molutils.mol_without_h(mol)))
        info["atoms_used_for_surface"] = int(mol.GetNumAtoms())

    with steps.run("molecular surface (PyMOL)"):
        # PyMOL names the object after the parent folder; a fixed, plain name avoids clashes with
        # selection keywords or numeric names.
        tmp = mol_dir / "molsurf"
        tmp.mkdir()
        mesh = raw(molutils.mol2surfmesh)(mol, method="pymol", tempPath=(tmp / "surface.obj").as_posix())
        shutil.rmtree(tmp, ignore_errors=True)
        if mesh is None or len(mesh.vertices) == 0:
            raise StepError("PyMOL returned an empty surface")
        np.savez_compressed(
            f("_surface.npz"),
            vert=np.asarray(mesh.vertices, np.float32),
            face=np.asarray(mesh.faces, np.int32),
            atom_xyz=np.asarray(mol.GetConformer().GetPositions(), np.float32),
            atom_symbol=np.array([a.GetSymbol() for a in mol.GetAtoms()]),
            bonds=np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], np.int32).reshape(-1, 2),
        )
        mesh.export(f("_surface.glb"))
        info.update(surface_vertices=int(len(mesh.vertices)), surface_faces=int(len(mesh.faces)))

    with steps.run("UV atlas (Open3D)"):
        uv_glb = f("_pymolsurf_with_uv.glb")
        mesh_uv = raw(molutils.mesh2uv)(mesh, config=["open3d", args.o3d_size, args.o3d_gutter, 1],
                                        glbsavePath=uv_glb, pngsavePath=False)
        if mesh_uv is None or len(mesh_uv.vertices) == 0:
            raise StepError("UV atlas generation returned an empty mesh")
        draw_uv_atlas(mesh_uv, f("_uvatlas.png"))

    with steps.run("UV repack + bake (Blender)"):
        objamage, _vert, _face, _uv, meta = raw(omgutil.objamage_encoderv2)(uv_glb)
        molmage = objamage["combined"]
        info["baked_resolution"] = int(molmage.shape[0])
        if args.resolution < molmage.shape[0]:
            molmage = omgutil.downsample_omg(molmage, factor=molmage.shape[0] // args.resolution)["omg_down_star"]
            occ = molmage[..., 3] > 0.5
            molmage[~occ, :] = 0
        np.save(f(f"_molmage{args.resolution}.npy"), molmage)
        os.remove(uv_glb)  # the official script deletes it as well
        info["uv_patches"] = {k: int(v) for k, v in (meta or {}).items()}

    return mol_id, mol_dir, molmage


# --------------------------------------------------------------------------- decode
def describe_molmage(molmage, info):
    occ = molmage[..., 3] > 0.5
    pos = molmage[..., :3][occ]
    info.update(molmage_shape=list(molmage.shape), occupancy_ratio=round(float(occ.mean()), 4),
                position_range=[round(float(pos.min()), 4), round(float(pos.max()), 4)])
    if not (-0.01 <= info["position_range"][0] and info["position_range"][1] <= 1.01):
        info["warning"] = f"positions span {info['position_range']}, MolMage expects [0, 1]"
        log(f"  warning: {info['warning']}")


def decode_mesh(molmage, mol_id, mol_dir, args, mm, steps, info):
    _molutils, omgutil, _visutil = mm
    f = namer(mol_dir, mol_id)
    describe_molmage(molmage, info)

    with steps.run("decode mesh"):
        omg = omgutil.tensor2omg(molmage)
        # 4-channel MolMages have no normal map: tensor2omg fills it with zeros, which meshing_objamage
        # turns into the constant direction (-1, -1, -1) and orients every face against it, mixing the
        # winding inside a patch. Same vertices and faces without that flip, so the interactive viewer
        # gets clean smooth shading. The GLB below is the official one.
        ret = omgutil.meshing_objamage(dict(omg, objnormal=None))
        if len(ret["face"]) == 0:
            raise StepError("The MolMage decodes to an empty mesh")
        np.savez_compressed(
            f("_mmg_mesh.npz"),
            vert=np.asarray(ret["vert"], np.float32),
            face=np.asarray(ret["face"], np.int32),
            vert_color=np.asarray(ret["segcolor"], np.float32)[:, :3],
            seg_map=np.asarray(omg["segmentation"])[..., 0].astype(np.int32),
            segcolor_map=np.asarray(omg["segcolor"], np.float32),
        )
        info.update(decoded_vertices=int(len(ret["vert"])), decoded_faces=int(len(ret["face"])),
                    n_patches=int(np.asarray(omg["segmentation"]).max()))

    if not args.no_glb:
        with steps.run("export GLB (Blender)"):
            omgutil.objamage_decoderv2(molmage, f("_mmg.blend"))  # official: _mmg.blend + _mmg.glb
            try:  # extra: the same mesh with the patch colours as texture
                omgutil.objamage_decoderv2(molmage, f("_mmg_patches.blend"), geometry_only=True,
                                           render_mode="segcolor")
            except Exception as exc:
                log(f"  (patch-coloured GLB skipped: {exc})")
            if os.path.exists(f("_mmg_patches.blend")):
                os.remove(f("_mmg_patches.blend"))


def load(args, mm, steps, info):
    src = Path(args.npy)
    with steps.run("read MolMage"):
        molmage = normalize_molmage(np.load(src))  # load first: the output folder is recreated below
    stem = re.sub(r"_molmage\d*$", "", src.stem)
    mol_id = sanitize_id(args.mol_id) or sanitize_id(stem) or "molmage"
    mol_dir = fresh_dir(Path(args.output_dir).resolve() / f"{mol_id}_decode")
    info.update(kind="molmage", input=src.name, mol_id=mol_id, out_dir=mol_dir.as_posix())
    np.save(namer(mol_dir, mol_id)(f"_molmage{molmage.shape[0]}.npy"), molmage)
    return mol_id, mol_dir, molmage


# --------------------------------------------------------------------------- render
def render(args, mm, steps, info):
    _molutils, omgutil, visutil = mm
    import bpy
    from xgutils import bpyutil

    mol_dir = Path(args.result_dir).resolve()
    meta = json.loads((mol_dir / "result.json").read_text())
    keep = {k: v for k, v in meta.items()
            if k not in ("status", "command", "error", "failed_step", "timings_s", "total_s", "versions", "render")}
    info.update(keep)
    info["encode_timings_s"] = meta.get("encode_timings_s", meta.get("timings_s", {}))
    info["encode_total_s"] = meta.get("encode_total_s", meta.get("total_s"))
    mol_id = meta["mol_id"]
    f = namer(mol_dir, mol_id)
    found = sorted(mol_dir.glob(f"{mol_id}_molmage*.npy"))
    if not found:
        raise StepError(f"No MolMage .npy in {mol_dir}")
    molmage = normalize_molmage(np.load(found[0]))
    shots = mol_dir / "snapshots"
    if shots.exists():
        shutil.rmtree(shots)
    shots.mkdir()

    with steps.run("official preview (omgutil.preview_omg)"):
        vomg, rdimg, _ = omgutil.preview_omg(
            molmage, resolution=args.size,
            render_kwargs=dict(shadow_catcher=True, samples=args.samples, camera_position=None))
        visutil.saveImg(f("_mmg.png"), vomg)  # official: [position | patches | render]
        frames = [np.asarray(rdimg, np.float32)]

    if args.views > 1:
        with steps.run(f"turntable x{args.views} (Blender Cycles)"):
            obj = bpy.data.objects.get("objamage_decoded")
            if obj is None:
                raise StepError("decoded object not found in the Blender scene")
            for k in range(1, args.views):
                obj.rotation_euler = (0.0, 0.0, 2.0 * np.pi * k / args.views)
                img = bpyutil.render_scene(obj=obj, resolution=(args.size,) * 2, samples=args.samples,
                                           shadow_catcher=True)
                frames.append(np.asarray(img, np.float32))
            obj.rotation_euler = (0.0, 0.0, 0.0)

    names = []
    for k, frame in enumerate(frames):
        name = f"view_{round(360 * k / args.views):03d}.png"
        visutil.saveImg((shots / name).as_posix(), frame)
        names.append(name)
    visutil.saveImg(f("_mmg_views.png"), tile(frames, cols=min(4, len(frames))))
    info["render"] = dict(views=args.views, size=args.size, samples=args.samples, frames=names,
                          seconds=round(sum(steps.timings.values()), 1))


# --------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    result_json = Path(args.result_json) if args.result_json else out_root / "last_result.json"
    if result_json.exists():
        result_json.unlink()
    tmp_dir = out_root / ".tmp"
    os.environ["MOLMAGE_TMP"] = tmp_dir.as_posix()  # bpyutil.render_scene writes its temp PNGs here

    steps = Steps()
    info = {"status": "running", "command": args.command}
    t0 = time.time()
    code = 0
    try:
        with steps.run("import MolMage"):
            mm = import_molmage(args.molmage_root)
        if args.command == "encode":
            mol_id, mol_dir, molmage = encode(args, mm, steps, info)
            decode_mesh(molmage, mol_id, mol_dir, args, mm, steps, info)
        elif args.command == "load":
            mol_id, mol_dir, molmage = load(args, mm, steps, info)
            decode_mesh(molmage, mol_id, mol_dir, args, mm, steps, info)
        else:
            render(args, mm, steps, info)
        info["status"] = "ok"
    except StepError as exc:
        traceback.print_exc()
        info.update(status="error", error=str(exc), failed_step=steps.current)
        log(f"FAILED: {exc}")
        code = 2
    except Exception as exc:
        traceback.print_exc()
        info.update(status="error", error=f"{type(exc).__name__}: {exc}", failed_step=steps.current)
        log(f"FAILED: {exc}")
        code = 1
    finally:
        info["timings_s"] = steps.timings
        info["total_s"] = round(time.time() - t0, 1)
        info["versions"] = versions()
        out_dir = Path(info["out_dir"]) if info.get("out_dir") else None
        if out_dir is not None and out_dir.is_dir():
            info["files"] = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
                                   if p.is_file() and p.name not in ("result.json", "run.log"))
            if info["status"] == "ok" or args.command != "render":  # a failed render keeps the old record
                (out_dir / "result.json").write_text(json.dumps(info, indent=2, default=str))
        result_json.write_text(json.dumps(info, indent=2, default=str))
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if code == 0:
        log(f"OK -> {info.get('out_dir')} ({info['total_s']} s)")
    return code


if __name__ == "__main__":
    sys.exit(main())
