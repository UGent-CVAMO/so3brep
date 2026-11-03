"""Run configuration: YAML files with a `base:` chain, environment variables and command-line overrides.

A config file holds only what differs from its base, e.g.

    base: ../base/fusion360.yaml
    model: {message: concat}

Strings may contain `${VAR}` or `${VAR:-default}`; `${SO3BREP_DATA:-data}` is the usual data root.
Overrides use dotted keys: `--set data.batch_size=16 train.patience=10`.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclass
class DataConfig:
    dataset: str = "fusion360"
    root: str = "data/fusion360"
    graphs: str = "so3brep"           # folder under <root>/graphs and <root>/graphs_rotated
    batch_size: int = 64
    num_workers: int = 6
    train_subset: int | None = None   # keep this many training parts (seeded by the run seed)


@dataclass
class TrainConfig:
    epochs: int = 300
    patience: int = 25                # early stopping on validation loss
    min_delta: float = 1e-3
    lr: float = 1e-3                  # AdamW peak learning rate
    weight_decay: float = 0.05
    warmup_epochs: int = 5            # linear warm-up, then cosine decay to lr / 50 at `epochs`


@dataclass
class Config:
    name: str = "run"
    data: DataConfig = field(default_factory=DataConfig)
    model: dict = field(default_factory=dict)  # SO3brep options; empty = the paper model
    train: TrainConfig = field(default_factory=TrainConfig)
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    out: str = "runs"

    def to_dict(self) -> dict:
        return asdict(self)


def _expand(value):
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    """The file merged over its `base:` chain; `name` is not inherited (it defaults to the file stem)."""
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    raw.setdefault("name", path.stem)
    if not base:
        return raw
    inherited = _read((path.parent / base).resolve())
    inherited.pop("name", None)
    return _merge(inherited, raw)


def _set(tree: dict, dotted: str, text: str) -> None:
    *parents, leaf = dotted.split(".")
    for p in parents:
        tree = tree.setdefault(p, {})
    tree[leaf] = yaml.safe_load(text)


def load(path: str | Path | None = None, overrides: list[str] = ()) -> Config:
    tree = _read(Path(path)) if path else {}
    for item in overrides:
        key, _, value = item.partition("=")
        _set(tree, key, value)
    tree = _expand(tree)
    return Config(
        name=tree.get("name", "run"),
        data=DataConfig(**tree.get("data", {})),
        model=tree.get("model", {}),
        train=TrainConfig(**tree.get("train", {})),
        seeds=tree.get("seeds", [42, 43, 44, 45, 46]),
        out=tree.get("out", "runs"),
    )
