"""Print the paper's tables from finished runs (mean ± std over seeds).

    python -m eval.tables runs

Reads runs/<group>/<name>/seed<k>/{config.yaml, metrics.json} as written by the reproduce scripts:
benchmark (Tabs. 1-2), message_functions (Tab. 3), ablations (Fig. 4), data_efficiency (Fig. 3).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

SEGMENTATION = ("mfcad", "mftrcad", "fusion360")
CLASSIFICATION = ("solidletters", "bendfm")


def load_runs(root: Path) -> dict[str, dict[str, list[dict]]]:
    """{group: {run name: [one record per seed]}}."""
    runs: dict = defaultdict(lambda: defaultdict(list))
    for metrics in sorted(root.glob("*/*/seed*/metrics.json")):
        cfg = yaml.safe_load((metrics.parent / "config.yaml").read_text())
        runs[metrics.parents[2].name][metrics.parents[1].name].append(json.loads(metrics.read_text()) | {"cfg": cfg})
    return runs


def stat(records: list[dict], split: str, key: str) -> str:
    vals = [100 * r[split][key] for r in records if key in r.get(split, {})]
    return f"{statistics.mean(vals):6.2f} ± {statistics.pstdev(vals):.2f}" if vals else "-"


def table(title: str, header: list[str], lines: list[list]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *lines)]
    print(f"\n{title}")
    for line in [header] + lines:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths)))


def benchmark(runs: dict) -> None:
    by_ds = {recs[0]["cfg"]["data"]["dataset"]: recs for recs in runs.values()}
    table("Tab. 1  Face segmentation", ["dataset", "seeds", "acc", "rot. acc", "mIoU", "rot. mIoU"],
          [[d, len(r), stat(r, "original", "acc"), stat(r, "rotated", "acc"), stat(r, "original", "iou"),
            stat(r, "rotated", "iou")] for d, r in by_ds.items() if d in SEGMENTATION])
    table("Tab. 2  Classification", ["dataset", "seeds", "acc", "rot. acc"],
          [[d, len(r), stat(r, "original", "acc"), stat(r, "rotated", "acc")]
           for d, r in by_ds.items() if d in CLASSIFICATION])


def message_functions(runs: dict) -> None:
    lines = []
    for name, r in sorted(runs.items()):
        cfg = r[0]["cfg"]
        parts = cfg["data"].get("train_subset") or "full"
        lines.append([cfg["data"]["dataset"], cfg["model"].get("message", "cg"), parts, len(r),
                      stat(r, "original", "acc"), stat(r, "original", "iou"),
                      stat(r, "rotated", "acc"), stat(r, "rotated", "iou")])
    table("Tab. 3  Message functions", ["dataset", "message", "train parts", "seeds", "acc", "mIoU", "rot. acc",
                                         "rot. mIoU"], lines)


def ablations(runs: dict) -> None:
    lines = [[name.split("_", 1)[1], len(r), stat(r, "original", "acc"), stat(r, "original", "iou")]
             for name, r in sorted(runs.items())]
    table("Fig. 4  Ablations (Fusion 360)", ["variant", "seeds", "acc", "mIoU"], lines)


def data_efficiency(runs: dict) -> None:
    lines = sorted(([r[0]["cfg"]["data"]["train_subset"], len(r), stat(r, "original", "acc")] for r in runs.values()),
                   key=lambda x: x[0])
    table("Fig. 3  MFCAD++ accuracy vs training parts", ["train parts", "seeds", "acc"], lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", type=Path, nargs="?", default=Path("runs"))
    runs = load_runs(p.parse_args(argv).runs)
    for group, show in (("benchmark", benchmark), ("message_functions", message_functions),
                        ("ablations", ablations), ("data_efficiency", data_efficiency)):
        if group in runs:
            show(runs[group])


if __name__ == "__main__":
    main()
