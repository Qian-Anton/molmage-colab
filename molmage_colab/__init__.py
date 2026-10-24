"""MolMage on Colab: SMILES / SDF -> MolMage -> 3D mesh + Blender snapshots.

The heavy lifting (RDKit, PyMOL, Open3D, Blender) runs in a separate Python 3.10 environment created
by :func:`setup`; this package drives it and shows the results in the notebook.

    import molmage_colab as mc
    mc.setup()
    result = mc.run("SMILES", smiles="CC(=O)Oc1ccccc1C(=O)O")
    mc.show_snapshots(views=4)
    mc.show_mesh(color_by="patch")
"""
from .display import download, show_mesh, show_molmage, show_snapshots, to_blender_frame
from .installer import setup
from .pipeline import Result, latest, render, run

__all__ = ["setup", "run", "render", "latest", "Result", "show_molmage", "show_snapshots", "show_mesh",
           "download", "to_blender_frame"]
__version__ = "0.1.0"
