"""Builds frontend/public/heart_model.glb from BodyParts3D anatomical meshes.

Source: BodyParts3D, (c) The Database Center for Life Science,
licensed under CC Attribution 4.0 International. https://lifesciencedb.jp/bp3d/

Each anatomical group becomes one named glTF node. Node names for vessels are the
same keys the ML targets and the frontend use ("lad", "lcx", "rca"), which is what
keeps model outputs and displayed anatomy in correspondence.

Usage (project root, venv active):  python -m tools.build_heart_model
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from trimesh.visual.material import PBRMaterial

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "tools" / ".cache" / "bodyparts3d"
OUTPUT_PATH = ROOT / "frontend" / "public" / "heart_model.glb"

BASE_URL = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/"
ZIP_NAME = "partof_BP3D_4.0_obj_99.zip"
ELEMENTS_NAME = "partof_element_parts.txt"
OBJ_DIR_IN_ZIP = "partof_BP3D_4.0_obj_99"

TARGET_HEIGHT = 2.0  # scene units; the whole heart is scaled to this height

# BodyParts3D: +X patient-left, -Y anterior, +Z superior (mm).
# three.js:   +X screen-right, +Z toward camera, +Y up -> anterior view by default.
BP3D_TO_THREE = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)


@dataclass(frozen=True)
class AnatomyGroup:
    node_name: str
    include: tuple[str, ...]           # FMA concept ids whose element meshes are included
    exclude: tuple[str, ...] = ()      # FMA concept ids removed from the selection
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    max_faces: int | None = None       # decimation budget; None keeps full resolution


CORONARY_VEINS = ("FMA4706", "FMA4707", "FMA4712", "FMA4713", "FMA4714")

GROUPS: tuple[AnatomyGroup, ...] = (
    AnatomyGroup("lad", ("FMA3862",), color=(0.75, 0.2, 0.2)),
    AnatomyGroup("lcx", ("FMA3895",), color=(0.75, 0.2, 0.2)),
    AnatomyGroup("rca", ("FMA50039",), color=(0.75, 0.2, 0.2)),
    AnatomyGroup("left_main", ("FMA3855",), color=(0.75, 0.2, 0.2)),
    AnatomyGroup("coronary_veins", CORONARY_VEINS, color=(0.25, 0.3, 0.6), max_faces=6000),
    AnatomyGroup("aorta", ("FMA3736", "FMA3768"), color=(0.85, 0.45, 0.45)),
    AnatomyGroup("pulmonary_trunk", ("FMA8612",), color=(0.45, 0.5, 0.75)),
    AnatomyGroup(
        "myocardium",
        ("FMA7088",),
        exclude=("FMA50039", "FMA50040") + CORONARY_VEINS,
        color=(0.62, 0.22, 0.25),
        max_faces=30000,
    ),
)


def ensure_source_files() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (ZIP_NAME, ELEMENTS_NAME):
        target = CACHE_DIR / name
        if not target.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, target)


def load_element_index() -> pd.DataFrame:
    df = pd.read_csv(CACHE_DIR / ELEMENTS_NAME, sep="\t")
    df.columns = ["concept_id", "name", "element_id"]
    return df


def element_ids_for(index: pd.DataFrame, group: AnatomyGroup) -> list[str]:
    def ids(concepts: tuple[str, ...]) -> set[str]:
        return set(index.loc[index.concept_id.isin(concepts), "element_id"])

    selected = ids(group.include) - ids(group.exclude)
    if not selected:
        raise ValueError(f"No meshes found for group {group.node_name!r}")
    return sorted(selected)


def load_group_mesh(archive: zipfile.ZipFile, element_ids: list[str]) -> trimesh.Trimesh:
    parts = [
        trimesh.load(io.BytesIO(archive.read(f"{OBJ_DIR_IN_ZIP}/{eid}.obj")), file_type="obj", force="mesh")
        for eid in element_ids
    ]
    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    return mesh


def decimate(mesh: trimesh.Trimesh, max_faces: int | None) -> trimesh.Trimesh:
    if max_faces is None or len(mesh.faces) <= max_faces:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=max_faces)


def normalisation_transform(meshes: list[trimesh.Trimesh]) -> np.ndarray:
    """Re-orients to three.js axes, centres the heart on the origin and scales to TARGET_HEIGHT."""
    bounds = np.vstack([m.bounds for m in meshes])
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    centre = (lo + hi) / 2
    height_mm = hi[2] - lo[2]  # superior axis before re-orientation

    to_origin = trimesh.transformations.translation_matrix(-centre)
    scale = trimesh.transformations.scale_matrix(TARGET_HEIGHT / height_mm)
    return BP3D_TO_THREE @ scale @ to_origin


def build() -> None:
    ensure_source_files()
    index = load_element_index()

    with zipfile.ZipFile(CACHE_DIR / ZIP_NAME) as archive:
        meshes = {g.node_name: load_group_mesh(archive, element_ids_for(index, g)) for g in GROUPS}

    transform = normalisation_transform(list(meshes.values()))
    scene = trimesh.Scene()
    for group in GROUPS:
        mesh = decimate(meshes[group.node_name], group.max_faces)
        mesh.apply_transform(transform)
        mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(
                name=group.node_name,
                baseColorFactor=[*group.color, 1.0],
                roughnessFactor=0.6,
                metallicFactor=0.0,
            )
        )
        scene.add_geometry(mesh, node_name=group.node_name, geom_name=group.node_name)
        print(f"  {group.node_name:<16} {len(mesh.faces):>7} faces")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(scene.export(file_type="glb"))
    total = sum(len(g.faces) for g in scene.geometry.values())
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB, {total} faces)")


if __name__ == "__main__":
    build()
