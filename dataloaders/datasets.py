"""Datasets of preprocessed B-rep face graphs (DGL `.bin` files, written by `pipeline.preprocess`).

    <root>/graphs/<graphs>/<stem>.bin                   aligned parts
    <root>/graphs_rotated/<graphs>/<stem>_rotated.bin   rotated test parts
    <root>/{train,val,test}.txt                         one stem per line

A graph has one node type `face` and one edge type `adj` (both directions of every adjacency).
Node data: `x_local` (UV grid in the face frame, N x 10 x 10 x 7), `center_nuv` (N x 12: centre,
normal, u-tangent, v-tangent), `face_feat` (N x 7), `vision_grids` (ray grids, N x 6 x 12 x 6),
`y` (face labels). Edge data: `x` (curve samples, E x 10 x 12), `edge_feat` (E x 10), `edge_mid`.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

ADJ = ("face", "adj", "face")


@dataclass(frozen=True)
class DatasetInfo:
    num_classes: int
    segmentation: bool


DATASETS = {
    "mfcad": DatasetInfo(25, True),        # MFCAD++ machining features
    "mftrcad": DatasetInfo(27, True),      # MFTRCAD machining features
    "fusion360": DatasetInfo(8, True),     # Fusion 360 Gallery modelling operations
    "solidletters": DatasetInfo(26, False),
    "bendfm": DatasetInfo(2, False),       # BenDFM: bendable without tool collision
}

_LETTERS = {c: i for i, c in enumerate(string.ascii_lowercase)}


def letter_label(stem: str) -> int:
    """SolidLetters stems start with the letter, e.g. `a_ABeeZee_upper`."""
    return _LETTERS[stem[0].lower()]


def _strip_rotated(stem: str) -> str:
    return stem[: -len("_rotated")] if stem.endswith("_rotated") else stem


def read_split(root: Path, split: str) -> set[str]:
    path = Path(root) / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"split file {path} not found")
    return {s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()}


# ----------------------------------------------------------------------------- labels

class _BenDFMLabels:
    """Per-part `y_tool_collision` from `<labels_dir>/<stem>_labels.json`."""

    def __init__(self, labels_dir: Path, key: str = "y_tool_collision"):
        self.dir, self.key = Path(labels_dir), key
        self._cache: dict[str, int] = {}

    def __call__(self, stem: str) -> int:
        stem = _strip_rotated(stem)
        if stem not in self._cache:
            path = self.dir / f"{stem}_labels.json"
            raw = json.loads(path.read_text()).get(self.key) if path.is_file() else None
            self._cache[stem] = -1 if raw is None else int(float(raw) > 0)
        return self._cache[stem]


class _MFTRCADLabels:
    """Per-face labels from `<labels_dir>/<stem>.json` ({"cls": {face index: label}})."""

    def __init__(self, labels_dir: Path):
        self.files = {p.stem: p for p in Path(labels_dir).glob("*.json")}

    def __call__(self, stem: str, n_faces: int) -> torch.Tensor:
        path = self.files.get(_strip_rotated(stem))
        cls = json.loads(path.read_text()).get("cls", {}) if path else {}
        return torch.tensor([cls.get(str(i), -1) for i in range(n_faces)], dtype=torch.long)


# ----------------------------------------------------------------------------- dataset

class BRepGraphs(Dataset):
    """Graphs of one split. Items: {"graph": DGL graph, "filename", "label" (classification only)}."""

    def __init__(self, dataset: str, graph_dir: Path, *, stems: set[str], labels_dir: Path):
        self.dataset = dataset
        self.info = DATASETS[dataset]
        self.root = Path(graph_dir)
        self.file_paths = [p for p in sorted(self.root.rglob("*.bin"))
                           if _strip_rotated(p.relative_to(self.root).with_suffix("").as_posix()) in stems]
        self._bendfm = _BenDFMLabels(labels_dir) if dataset == "bendfm" else None
        self._mftrcad = _MFTRCADLabels(labels_dir) if dataset == "mftrcad" else None

    def part_label(self, path: Path) -> int:
        return letter_label(path.stem) if self.dataset == "solidletters" else self._bendfm(path.stem)

    def subset(self, n: int, seed: int) -> None:
        """Keep `n` training parts at random; stratified by class for classification."""
        if n >= len(self.file_paths):
            return
        if self.info.segmentation:
            keep = np.random.RandomState(seed).choice(len(self.file_paths), size=n, replace=False)
        else:
            from sklearn.model_selection import train_test_split
            labels = [self.part_label(p) for p in self.file_paths]
            keep, _ = train_test_split(np.arange(len(labels)), train_size=n, stratify=labels, random_state=seed)
        self.file_paths = [self.file_paths[i] for i in np.sort(keep)]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.file_paths[idx]
        graph = load_graphs(str(path))[0][0]
        faces = graph.nodes["face"].data
        for k in list(faces.keys()):
            faces[k] = faces[k].long() if k == "y" else faces[k].float()
        edges = graph.edges[ADJ].data
        for k in list(edges.keys()):
            edges[k] = edges[k].float()
        if self.dataset == "mftrcad":
            faces["y"] = self._mftrcad(path.stem, graph.num_nodes("face"))
        item = {"graph": graph, "filename": path.stem}
        if not self.info.segmentation:
            item["label"] = torch.tensor([self.part_label(path)], dtype=torch.long)
        return item


# ----------------------------------------------------------------------------- PyG conversion

def to_pyg(graph, label: torch.Tensor | None = None) -> Data:
    """One DGL face graph as the PyG `Data` SO3brep reads. Faces without a neighbour get a self-loop."""
    faces, edges = graph.nodes["face"].data, graph.edges[ADJ].data
    src, dst = (t.long() for t in graph.edges(etype=ADJ))
    n = faces["x_local"].shape[0]
    connected = torch.zeros(n, dtype=torch.bool)
    connected[src] = connected[dst] = True
    isolated = torch.where(~connected)[0]
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([isolated, isolated])], dim=1)

    def pad(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, t.new_zeros(isolated.numel(), *t.shape[1:])])

    if label is None:
        label = faces["y"].long() if "y" in faces else torch.zeros(n, dtype=torch.long)
    return Data(
        x_grid_local=faces["x_local"].permute(0, 3, 1, 2).contiguous(),
        vision_grids=faces["vision_grids"].permute(0, 3, 1, 2).contiguous(),
        face_feat=faces["face_feat"],
        center_nuv=faces["center_nuv"],
        pos=faces["center_nuv"][:, :3],
        edge_index=edge_index,
        edge_attr=pad(edges["edge_feat"]),
        edge_mid=pad(edges["edge_mid"]),
        edge_curve_pts=pad(edges["x"]),
        num_nodes=n,
        y=label,
    )


def collate_pyg(batch: list[dict]) -> dict:
    """A list of dataset items -> {"graph": PyG Batch, "filename", "label" (classification only)}."""
    classification = "label" in batch[0]
    datas = [to_pyg(item["graph"], item["label"].squeeze() if classification else None) for item in batch]
    out = {"graph": Batch.from_data_list(datas), "filename": [s["filename"] for s in batch]}
    if classification:
        out["label"] = torch.stack([d.y for d in datas])
    return out
