"""STEP files to DGL face-adjacency graphs.

Builds one graph per part: nodes are B-rep faces (UV grid in the face frame, surface
attributes, local frame, ray grids), edges are face adjacencies (curve samples,
curve attributes, midpoint). Parts are normalised to unit radius; ``--rotate`` also
applies a random rotation (graphs for the rotated test set).

    python -m pipeline.preprocess --root DATA/mfcad++ --dataset mfcad++ --split all
    python -m pipeline.preprocess --root DATA/mfcad++ --dataset mfcad++ --split test --rotate

Layout under ``--out_root`` (default ``--root``): ``graphs/<folder>/<id>.bin``,
or ``graphs_rotated/<folder>/<id>_rotated.bin`` and
``step_rotated/<id>_rotated.stp`` with ``--rotate``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
from OCC.Extend.TopologyUtils import TopologyExplorer
from tqdm import tqdm

from pipeline import geometry, occ
from pipeline.ray_casting import ANGULAR_DEFLECTION, LINEAR_DEFLECTION, MeshRayCaster, cast_from_faces

UV_SAMPLES = 10
CURVE_SAMPLES = 10
VISION_ELEV, VISION_AZIM = 6, 12
MAX_RAY_DIST = 2.0  # twice the unit radius
TIMEOUT = 300  # seconds per part
SPLITS = ("train", "val", "test")
DEFAULT_FOLDER = "so3brep"

# SolidLetters fonts that are excluded (they render as symbols or are broken).
INVALID_FONTS = frozenset([
    "Bokor", "Lao Muang Khong", "Lao Sans Pro", "MS Outlook", "Catamaran Black",
    "Dubai", "HoloLens MDL2 Assets", "Lao Muang Don", "Oxanium Medium",
    "Rounded Mplus 1c", "Moul Pali", "Noto Sans Tamil", "Webdings", "Armata",
    "Koulen", "Yinmar", "Ponnala", "Chenla", "Lohit Devanagari", "Metal",
    "MS Office Symbol", "Cormorant Garamond Medium", "Chiller", "Give You Glory",
    "Hind Vadodara Light", "Libre Barcode 39 Extended", "Myanmar Sans Pro",
    "Scheherazade", "Segoe MDL2 Assets", "Siemreap", "Signika SemiBold",
    "Taprom", "Times New Roman TUR", "Playfair Display SC Black", "Poppins Thin",
    "Raleway Dots", "Raleway Thin", "Spectral SC ExtraLight", "Txt", "Uchen",
    "Almarai ExtraBold", "Fasthand", "Exo", "Freckle Face", "Montserrat Light",
    "Inter", "MS Reference Specialty", "Preah Vihear", "Sitara",
    "Barkerville Old Face", "Bodoni MT", "HoloLens MDL2 Assests",
    "Libre Barcode 39", "Lohit Tamil", "Marlett", "MS outlook",
    "MS office Symbol Semilight", "MS office symbol regular",
    "Ms office symbol extralight", "Ms Reference speciality", "Symbol",
    "Wingdings", "Souliyo Unicode", "Aguafina Script", "Yantramanav Black",
])


@dataclass(frozen=True)
class Task:
    """One part to convert."""
    step: Path
    step_root: Path
    out_dir: Path
    dataset: str
    rotate: bool
    rotated_step_dir: Optional[Path] = None

    @property
    def part_id(self) -> str:
        """Path of the STEP file below the step root, without extension."""
        return self.step.relative_to(self.step_root).with_suffix("").as_posix()

    @property
    def graph_path(self) -> Path:
        return self.out_dir / (self.part_id + ("_rotated" if self.rotate else "") + ".bin")


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def _vision(rays: dict) -> np.ndarray:
    """Ray grids (F, n_el, n_az, 6): hit, distance and normal dot for the outward and inward hemisphere."""
    return np.stack([rays["occ"][:, 0], rays["dist"][:, 0], rays["dot"][:, 0],
                     rays["occ"][:, 1], rays["dist"][:, 1], rays["dot"][:, 1]], axis=-1)


def build_graph(task: Task, shape, mesh=None, frames=None) -> dgl.DGLGraph:
    """Build the graph of one normalised shape."""
    faces, adjacency = occ.face_adjacency(shape)
    n_faces = len(faces)
    if n_faces == 0 or not adjacency:
        raise ValueError("no faces" if n_faces == 0 else "no face adjacency")
    keys = sorted(adjacency)
    caster = MeshRayCaster(*mesh) if mesh is not None else MeshRayCaster.from_shape(shape)

    feats = [geometry.face_features(f, UV_SAMPLES) for f in faces]
    if frames is None:
        frames = [(f["center"], f["axes"]) for f in feats]
    rays = cast_from_faces(
        caster, np.array([c for c, _ in frames]), [a for _, a in frames], VISION_ELEV, VISION_AZIM, MAX_RAY_DIST,
    )
    node = {k: np.stack([f[k] for f in feats]) for k in ("x_local", "face_feat", "center_nuv")}
    node["vision_grids"] = _vision(rays)
    if task.dataset in geometry.SEGMENTATION_DATASETS:
        node["y"] = geometry.face_labels(task.step, task.dataset, n_faces)

    edge = _edge_data(shape, adjacency, keys)
    src = [k[0] for k in keys]
    dst = [k[1] for k in keys]
    graph = dgl.heterograph(
        {("face", "adj", "face"): (torch.tensor(src + dst), torch.tensor(dst + src))},
        num_nodes_dict={"face": n_faces},
    )
    for name, value in node.items():
        graph.ndata[name] = torch.from_numpy(np.ascontiguousarray(value))
    for name, value in edge.items():
        t = torch.from_numpy(value)
        graph.edata[name] = torch.cat([t, t])
    return graph


def _edge_data(shape, adjacency: dict, keys: Sequence[Tuple[int, int]]) -> dict:
    """Per-edge arrays (one row per key, in key order); the caller duplicates them."""
    topo = TopologyExplorer(shape, ignore_orientation=True)
    faces_of_edge = {}
    grids, attrs, mids = [], [], []
    for key in keys:
        edge = adjacency[key]
        h = hash(edge)
        if h not in faces_of_edge:
            try:
                faces_of_edge[h] = list(topo.faces_from_edge(edge))
            except Exception:
                faces_of_edge[h] = []
        grid, convexity = geometry.edge_samples(edge, faces_of_edge[h], CURVE_SAMPLES)
        mid = geometry.edge_midpoint(edge)
        grids.append(grid)
        attrs.append(geometry.edge_attributes(edge, convexity))
        mids.append(mid)
    n = len(keys)
    out = {
        "x": np.array(grids).reshape(n, CURVE_SAMPLES, 12),
        "edge_mid": np.array(mids).reshape(n, 3),
    }
    if n:
        out["edge_feat"] = np.array(attrs, dtype=np.float64)
    return out


# --------------------------------------------------------------------------
# One part
# --------------------------------------------------------------------------

def process_part(task: Task) -> Tuple[bool, str]:
    """Convert one STEP file and write its graph; returns ``(ok, message)``."""
    shape = occ.scale_to_unit_radius(occ.load_body(task.step))
    mesh = frames = None
    if task.rotate:
        vertices, triangles, face_ids = occ.tessellate(shape, LINEAR_DEFLECTION, ANGULAR_DEFLECTION)
        mesh = (vertices, triangles, face_ids)
        frames = [
            geometry.local_frame(occ.strip_location(f))
            for f in TopologyExplorer(shape, ignore_orientation=True).faces()
        ]
        shape = occ.rotate(shape, occ.rotation_angles(task.part_id))
        occ.write_step(shape, task.rotated_step_dir / (task.part_id + "_rotated.stp"))
    graph = build_graph(task, shape, mesh, frames)
    task.graph_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = task.graph_path.with_name(task.graph_path.name + ".tmp")
    dgl.save_graphs(str(tmp), [graph])
    os.replace(tmp, task.graph_path)
    return True, ""


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

def read_ids(path: Path) -> set:
    """Non-empty lines of a text file."""
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def find_steps(step_dir: Path, dataset: str) -> List[Path]:
    """STEP files below ``step_dir`` (sorted), without unfolded states and bad fonts."""
    files = [f for f in step_dir.rglob("*") if f.is_file() and f.suffix.lower() in (".stp", ".step")]
    files = [f for f in files if "_unfolded" not in f.stem]
    if dataset == "solidletters":
        files = [f for f in files if len(f.stem.split("_")) > 1 and f.stem.split("_")[1] not in INVALID_FONTS]
    return sorted(files)


def select_steps(files: List[Path], step_dir: Path, ids: set) -> List[Path]:
    """Files whose stem or relative id is in ``ids``."""
    def wanted(f: Path) -> bool:
        return f.stem in ids or f.relative_to(step_dir).with_suffix("").as_posix() in ids
    return [f for f in files if wanted(f)]


def make_tasks(root: Path, out_root: Path, dataset: str, split: str, folder: str, rotate: bool,
               only: Optional[set]) -> List[Task]:
    """Tasks for the files of a split (``all`` for every split)."""
    step_dir = root / "step"
    ids = set()
    for name in (SPLITS if split == "all" else (split,)):
        ids |= read_ids(root / f"{name}.txt")
    if only is not None:
        ids &= only
    files = select_steps(find_steps(step_dir, dataset), step_dir, ids)
    out_dir = out_root / ("graphs_rotated" if rotate else "graphs") / folder
    rotated_steps = out_root / "step_rotated" if rotate else None
    return [Task(f, step_dir, out_dir, dataset, rotate, rotated_steps) for f in files]


def run(tasks: List[Task], workers: int, timeout: float, skip_existing: bool) -> Tuple[int, List[str]]:
    """Process tasks in killable worker processes; returns (n_ok, errors)."""
    if skip_existing:
        tasks = [t for t in tasks if not t.graph_path.exists()]
    errors, n_ok = [], 0
    results = occ.run_isolated(process_part, tasks, workers, timeout)
    for task, ok, message in tqdm(results, total=len(tasks), desc="graphs"):
        if ok:
            n_ok += 1
        else:
            errors.append(f"{task.part_id}: {message}")
            tqdm.write(f"[ERROR] {task.part_id}: {message}")
    return n_ok, errors


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="dataset folder with step/ and train|val|test.txt")
    p.add_argument("--dataset", required=True,
                   choices=["mfcad++", "fusion360", "solidletters", "mftrcad", "bendfm"])
    p.add_argument("--out_root", type=Path, default=None, help="where graphs/ is written (default: --root)")
    p.add_argument("--folder", default=DEFAULT_FOLDER, help="output sub-folder name")
    p.add_argument("--split", default="all", choices=[*SPLITS, "all"])
    p.add_argument("--rotate", action="store_true", help="random rotation (writes graphs_rotated/ and step_rotated/)")
    p.add_argument("--ids", type=Path, default=None, help="text file with part ids; restricts the split")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=TIMEOUT, help="seconds per part before it is killed")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args(argv)

    only = read_ids(args.ids) if args.ids else None
    tasks = make_tasks(args.root, args.out_root or args.root, args.dataset, args.split, args.folder,
                       args.rotate, only)
    print(f"{args.dataset}: {len(tasks)} parts, {args.workers} workers")
    n_ok, errors = run(tasks, args.workers, args.timeout, args.skip_existing)
    print(f"done: {n_ok} ok, {len(errors)} errors")


if __name__ == "__main__":
    main()
