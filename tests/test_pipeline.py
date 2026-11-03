"""STEP to graph on two small parts, then one epoch of training and evaluation."""

import json
from pathlib import Path

import dgl
import numpy as np
import pytest

pytest.importorskip("OCC")

from pipeline import preprocess  # noqa: E402
from train.config import load  # noqa: E402
from train.train import run  # noqa: E402

DATA = Path(__file__).parent / "test_data"
PARTS = [("mfcad++", "5290.step"), ("bendfm", "33655_f8991a06_4.stp")]


def convert(out: Path, dataset: str, name: str, rotate: bool = False):
    task = preprocess.Task(DATA / name, DATA, out / ("graphs_rotated" if rotate else "graphs") / "so3brep", dataset,
                           rotate, out / "step_rotated")
    assert preprocess.process_part(task) == (True, "")
    return dgl.load_graphs(str(task.graph_path))[0][0]


@pytest.mark.parametrize("dataset,name", PARTS)
def test_graph_fields(tmp_path, dataset, name):
    g = convert(tmp_path, dataset, name)
    n, m = g.num_nodes(), g.num_edges()
    shapes = {"x_local": (n, 10, 10, 7), "face_feat": (n, 7), "center_nuv": (n, 12), "vision_grids": (n, 6, 12, 6)}
    assert {k: tuple(g.ndata[k].shape) for k in shapes} == shapes
    assert {k: tuple(g.edata[k].shape) for k in ("x", "edge_feat", "edge_mid")} == \
        {"x": (m, 10, 12), "edge_feat": (m, 10), "edge_mid": (m, 3)}
    assert ("y" in g.ndata) == (dataset == "mfcad++")


def test_rotation_keeps_local_features(tmp_path):
    aligned = convert(tmp_path, *PARTS[1])
    rotated = convert(tmp_path, *PARTS[1], rotate=True)
    assert not np.allclose(aligned.ndata["center_nuv"], rotated.ndata["center_nuv"])
    for key in ("x_local", "vision_grids", "face_feat"):
        np.testing.assert_allclose(aligned.ndata[key], rotated.ndata[key], atol=1e-6)


def test_train_and_evaluate(tmp_path):
    root = tmp_path / "mfcad"
    convert(root, *PARTS[0])
    convert(root, *PARTS[0], rotate=True)
    for split in ("train", "val", "test"):
        (root / f"{split}.txt").write_text("5290\n")
    config = load(None, ["data.dataset=mfcad", f"data.root={root}", "data.batch_size=1", "data.num_workers=0",
                         "train.epochs=1", f"out={tmp_path / 'runs'}", "name=smoke"])
    run(config, seed=0)
    scores = json.loads((tmp_path / "runs" / "smoke" / "seed0" / "metrics.json").read_text())
    assert 0 <= scores["original"]["acc"] <= 1 and scores["original"] == scores["rotated"]
