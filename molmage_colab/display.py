"""Notebook output: every function shows one caption line and one row of panels."""
import shutil
from pathlib import Path

import numpy as np

from . import paths
from .pipeline import _require, render
from .ui import note

ATOM_COLORS = {"H": "#E8E8E8", "C": "#505050", "N": "#3050F8", "O": "#FF0D0D", "F": "#1FF01F", "P": "#FF8000",
               "S": "#E6C200", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094"}
ATOM_SIZES = {"H": 4}
MAX_ROW_WIDTH = 16.0  # inches


# --------------------------------------------------------------------------- captions
def caption(result):
    info = result.info
    shape = "×".join(map(str, info.get("molmage_shape", [])))
    parts = [result.name]
    if info.get("smiles"):
        parts.append(info["smiles"])
    elif info.get("kind") == "molmage":
        parts.append(f"from {info.get('input')}")
    parts += [f"MolMage {shape}", f"{info.get('n_patches')} UV patches",
              f"mesh {info.get('decoded_faces', 0):,} faces", f"{info.get('total_s')} s"]
    if info.get("filter_warnings"):
        parts.append("input filters skipped")
    if info.get("warning"):
        parts.append(info["warning"])
    return " · ".join(str(p) for p in parts)


# --------------------------------------------------------------------------- static rows
def _row(panels, height=3.4):
    """panels: list of (title, image, imshow kwargs) drawn side by side in one figure."""
    import matplotlib.pyplot as plt

    aspects = [img.shape[1] / img.shape[0] for _, img, _ in panels]
    height = min(height, MAX_ROW_WIDTH / sum(aspects))
    fig, axes = plt.subplots(1, len(panels), figsize=(height * sum(aspects), height + 0.35),
                             gridspec_kw=dict(width_ratios=aspects), squeeze=False)
    for ax, (title, img, kwargs) in zip(axes[0], panels):
        ax.imshow(img, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.9, wspace=0.04)
    plt.show()
    plt.close(fig)


def _load_png(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def show_molmage(result=None, with_caption=True):
    """Row: 2D structure | MolMage x, y, z as RGB | occupancy | UV patches."""
    result = _require(result)
    if result is None:
        return
    if with_caption:
        note(caption(result))
    m = result.molmage().astype(np.float32)
    occ = m[..., 3] > 0.5
    patches = np.load(result.file("_mmg_mesh.npz"))["segcolor_map"]
    pixel = dict(interpolation="nearest", vmin=0, vmax=1)
    panels = []
    if result.file("_2d.png").exists():
        panels.append(("2D structure", _load_png(result.file("_2d.png")), {}))
    panels += [
        ("MolMage  x, y, z → RGB", np.where(occ[..., None], np.clip(m[..., :3], 0, 1), 0.0), pixel),
        (f"occupancy · {occ.mean():.0%} of pixels", occ.astype(np.float32), dict(pixel, cmap="gray")),
        (f"UV patches · {result.info.get('n_patches')}", np.clip(patches, 0, 1), pixel),
    ]
    _row(panels)


def _crop_frames(frames, pad=0.06):
    """Crop all frames to the union of their non-transparent pixels (same box for every view)."""
    mask = np.any([f[..., 3] > 0.03 for f in frames], axis=0)
    if not mask.any():
        return frames
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    h, w = mask.shape
    side = max(rows[-1] - rows[0], cols[-1] - cols[0]) * (1 + 2 * pad)
    cy, cx = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
    top, left = int(max(0, cy - side / 2)), int(max(0, cx - side / 2))
    bottom, right = int(min(h, cy + side / 2)), int(min(w, cx + side / 2))
    return [f[top:bottom, left:right] for f in frames]


def show_snapshots(result=None, views=4, size=512, samples=32):
    """Row of Blender renders of the decoded mesh, rotated about the vertical axis.

    View 0° uses the camera of the official ``_mmg.png`` preview. Renders are cached per setting.
    """
    result = _require(result)
    if result is None:
        return
    done = result.info.get("render") or {}
    frames = [result.out_dir / "snapshots" / name for name in done.get("frames", [])]
    cached = (done.get("views") == views and done.get("size") == size and done.get("samples") == samples
              and frames and all(p.exists() for p in frames))
    if cached:
        note(f"{result.name} · {views} Blender snapshot{'s' if views > 1 else ''} (rendered earlier)")
    else:
        result = render(result, views=views, size=size, samples=samples)
        if result is None:
            return
        frames = [result.out_dir / "snapshots" / name for name in result.info["render"]["frames"]]
    images = _crop_frames([_load_png(p) for p in frames])
    titles = [f"{round(360 * k / views)}°" for k in range(views)]
    titles[0] += " · official view"
    _row([(t, img, {}) for t, img in zip(titles, images)], height=3.6)


# --------------------------------------------------------------------------- interactive row
def to_blender_frame(vert, ref=None):
    """Raw PyMOL / RDKit coordinates -> the normalised frame the MolMage is baked in.

    Blender's glTF importer maps (x, y, z) -> (x, -z, y); bpyutil.normalize_mesh then centres the
    bounding box of the surface (``ref``) and scales its longest side to 1.9999.
    """
    def swap(v):
        v = np.asarray(v, np.float64)
        return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)

    v, r = swap(vert), swap(vert if ref is None else ref)
    lo, hi = r.min(0), r.max(0)
    return (v - (lo + hi) / 2) * (1.9999 / (hi - lo).max())


def _mesh_trace(vert, face, rgb=None, color="#B0C4DE", opacity=1.0):
    import plotly.graph_objects as go

    kwargs = dict(x=vert[:, 0], y=vert[:, 1], z=vert[:, 2], i=face[:, 0], j=face[:, 1], k=face[:, 2],
                  opacity=opacity, flatshading=False, hoverinfo="skip", showscale=False,
                  lighting=dict(ambient=0.5, diffuse=0.8, specular=0.15, roughness=0.6, fresnel=0.1),
                  lightposition=dict(x=100, y=-200, z=300))
    if rgb is not None:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(int)
        kwargs["vertexcolor"] = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    else:
        kwargs["color"] = color
    return go.Mesh3d(**kwargs)


def _atom_traces(xyz, symbols, bonds):
    import plotly.graph_objects as go

    traces = []
    if len(bonds):
        seg = [[], [], []]  # one line trace, bonds separated by None gaps
        for a, b in bonds:
            for axis in range(3):
                seg[axis] += [float(xyz[a, axis]), float(xyz[b, axis]), None]
        traces.append(go.Scatter3d(x=seg[0], y=seg[1], z=seg[2], mode="lines", hoverinfo="skip",
                                   line=dict(color="#707070", width=5)))
    traces.append(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", text=list(symbols), hoverinfo="text",
        marker=dict(size=[ATOM_SIZES.get(s, 7) for s in symbols], color=[ATOM_COLORS.get(s, "#FF1493") for s in symbols],
                    line=dict(width=0))))
    return traces


def show_mesh(result=None, color_by="patch", atoms=True):
    """Interactive row: input conformer + PyMOL surface | mesh decoded from the MolMage.

    color_by="patch" colours the decoded mesh by UV patch (as in the snapshots); "position" uses the
    MolMage x, y, z colours on both meshes.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    result = _require(result)
    if result is None:
        return
    dec = np.load(result.file("_mmg_mesh.npz"))
    dv, df = dec["vert"], dec["face"]
    dec_rgb = dec["vert_color"] if color_by == "patch" else dv * 0.5 + 0.5
    decoded = _mesh_trace(dv, df, rgb=dec_rgb)
    surf_file = result.file("_surface.npz")
    if surf_file.exists():
        surf = np.load(surf_file)
        sv = to_blender_frame(surf["vert"])
        left = [_mesh_trace(sv, surf["face"], rgb=sv * 0.5 + 0.5 if color_by == "position" else None,
                            opacity=0.35 if atoms else 1.0)]
        if atoms:
            left += _atom_traces(to_blender_frame(surf["atom_xyz"], ref=surf["vert"]), surf["atom_symbol"],
                                 surf["bonds"])
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.02,
                            subplot_titles=(f"input · conformer + PyMOL surface ({len(surf['face']):,} faces)",
                                            f"decoded from the MolMage ({len(df):,} faces)"))
        for trace in left:
            fig.add_trace(trace, row=1, col=1)
        fig.add_trace(decoded, row=1, col=2)
    else:
        fig = go.Figure(decoded)
        fig.update_layout(title=dict(text=f"decoded from the MolMage ({len(df):,} faces)", x=0.5, font=dict(size=14)))
    axis = dict(visible=False)
    fig.update_scenes(aspectmode="data", xaxis=axis, yaxis=axis, zaxis=axis,
                      camera=dict(eye=dict(x=0.0, y=-1.9, z=0.6), up=dict(x=0, y=0, z=1)))
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=36, b=0), showlegend=False, paper_bgcolor="white")
    fig.show()


# --------------------------------------------------------------------------- files
def download(result=None, everything=False):
    """Zip the latest result folder (or every result) and download it in Colab."""
    if everything:
        target, name = paths.OUT_ROOT, "molmage_outputs"
    else:
        result = _require(result)
        if result is None:
            return
        target, name = result.out_dir, f"molmage_{result.name}"
    archive = Path(shutil.make_archive(str(paths.HOME / name), "zip", root_dir=target))
    note(f"{archive} · {archive.stat().st_size / 1e6:.1f} MB")
    try:
        from google.colab import files
    except ImportError:
        return
    files.download(str(archive))
