"""Per-face and per-edge geometric features and segmentation labels."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_REVERSED

from pipeline import occ

ZERO = 1e-6
MAX_AREA = 40.0  # parts are scaled to unit radius; larger face areas come from broken geometry
# Datasets whose face labels are stored in the graph. MFTRCAD labels are read at load time from
# labels/<id>.json (dataloaders.datasets).
SEGMENTATION_DATASETS = ("mfcad++", "fusion360")


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def _labels_from_step_names(path: Path) -> List[int]:
    """Integer face labels stored in the names of ADVANCED_FACE entities."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r"#\d+\s*=\s*ADVANCED_FACE\('([^']*)'\s*", text)
    return [int(name) for name in names if name.strip() and name.isdigit()]


def _labels_from_fusion360_seg(path: Path) -> Optional[List[int]]:
    """Labels of ``../seg/<stem>.seg``, one integer per line."""
    seg = path.parent.parent / "seg" / (path.stem + ".seg")
    if not seg.is_file():
        return None
    return [int(line) for line in seg.read_text().split("\n") if line.strip().lstrip("-").isdigit()]


def face_labels(path: Path, dataset: str, n_faces: int) -> np.ndarray:
    """Per-face labels (int64, -1 where unavailable) for a segmentation dataset."""
    if dataset == "fusion360":
        labels = _labels_from_fusion360_seg(path)
    else:
        labels = _labels_from_step_names(path)
    out = np.full(n_faces, -1, dtype=np.int64)
    if labels:
        n = min(n_faces, len(labels))
        out[:n] = labels[:n]
    return out


# --------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------

def local_frame(face) -> Tuple[np.ndarray, np.ndarray]:
    """Local frame at the middle of the surface parameter range.

    Returns the centre (3,) and axes (3, 3) with columns x (u tangent), y and
    z (surface normal, flipped for reversed faces).
    """
    surf = BRepAdaptor_Surface(face)
    u = (surf.FirstUParameter() + surf.LastUParameter()) * 0.5
    v = (surf.FirstVParameter() + surf.LastVParameter()) * 0.5
    props = BRepLProp_SLProps(surf, 1, ZERO * 1e-3)
    props.SetParameters(u, v)
    c = props.Value()
    center = np.array([c.X(), c.Y(), c.Z()])
    n = props.Normal()
    z = np.array([n.X(), n.Y(), n.Z()])
    norm = np.linalg.norm(z)
    if norm > ZERO:
        z /= norm
    if face.Orientation() == TopAbs_REVERSED:
        z = -z
    d = surf.Surface().DN(float(u), float(v), 1, 0)
    du = np.array([d.X(), d.Y(), d.Z()])
    x = du - np.dot(du, z) * z
    norm = np.linalg.norm(x)
    if norm < 1e-4:  # degenerate u tangent: the coordinate axis least aligned with the normal
        x = np.eye(3)[np.argmin(np.abs(z))]
        x = x - np.dot(x, z) * z
        norm = np.linalg.norm(x)
    x /= norm
    return center, np.stack([x, np.cross(z, x), z], axis=1)


def face_attributes(face) -> List[float]:
    """Seven face descriptors: plane, cylinder, cone, sphere, torus, area, rational NURBS."""
    try:
        surf = BRepAdaptor_Surface(face)
        kind = surf.GetType()
    except Exception:
        return [0.0] * 7
    try:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = abs(float(props.Mass()))
        if area > MAX_AREA:
            area = 0.0
    except Exception:
        area = 0.0
    rational = 0.0
    try:
        if kind == GeomAbs_BSplineSurface:
            s = surf.BSpline()
            rational = 1.0 if s.IsURational() or s.IsVRational() else 0.0
        elif kind == GeomAbs_BezierSurface:
            s = surf.Bezier()
            rational = 1.0 if s.IsURational() or s.IsVRational() else 0.0
    except Exception:
        pass
    flags = [kind == t for t in (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus)]
    return [float(f) for f in flags] + [area, rational]




def face_features(face, uv_samples: int) -> dict:
    """Per-face features: ``x_local`` (U, V, 7: xyz, normal and trim mask in the face frame),
    ``face_feat`` (7), ``center_nuv`` (12: centre, normal, u and v axes) and the frame itself."""
    face = occ.strip_location(face)
    points, normals, mask = occ.uv_grid(face, occ.trim_classifier(face), uv_samples, uv_samples)
    center, axes = local_frame(face)
    shape = (uv_samples, uv_samples, 3)
    local_pts = ((points.reshape(-1, 3) - center[None, :]) @ axes).reshape(shape)
    local_nrm = (normals.reshape(-1, 3) @ axes).reshape(shape)
    x_local = np.concatenate([local_pts, local_nrm, mask[..., None].astype(np.float64)], axis=-1)
    return {
        "x_local": x_local, "face_feat": np.asarray(face_attributes(face)),
        "center_nuv": np.concatenate([center, axes[:, 2], axes[:, 0], axes[:, 1]]),
        "center": center, "axes": axes,
    }


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------

def edge_midpoint(edge) -> np.ndarray:
    """Point of the edge curve at its parametric midpoint (zeros where undefined)."""
    try:
        curve = BRepAdaptor_Curve(edge)
        p = curve.Value(0.5 * (curve.FirstParameter() + curve.LastParameter()))
        return np.array([p.X(), p.Y(), p.Z()])
    except Exception:
        return np.zeros(3)


def edge_samples(edge, faces: list, n: int) -> Tuple[np.ndarray, Optional[str]]:
    """Samples (n, 12: point, tangent, left and right normal) and convexity of an edge.

    Zeros and the convexity from uniform parameters are returned when the
    arc-length sampling fails.
    """
    if not faces:
        return np.zeros((n, 12)), None
    try:
        s = occ.EdgeSamples(edge, faces, n, arclength=True)
        if s.good:
            grid = np.concatenate([s.points, s.tangents, s.left_normals, s.right_normals], axis=1)
            return grid.astype(np.float64), s.convexity()
    except Exception:
        pass
    try:
        s = occ.EdgeSamples(edge, faces, n, arclength=False)
        return np.zeros((n, 12)), s.convexity() if s.good else None
    except Exception:
        return np.zeros((n, 12)), None


def edge_attributes(edge, convexity: Optional[str]) -> List[float]:
    """Ten edge descriptors: concave, convex, smooth, length, circular, closed,
    elliptical, non-rational B-spline, rational B-spline, straight."""
    kind, rational = None, False
    try:
        curve = BRepAdaptor_Curve(edge)
        kind = curve.GetType()
        rational = curve.IsRational() if kind == GeomAbs_BSplineCurve else False
    except Exception:
        pass
    try:
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = float(props.Mass())
    except Exception:
        length = 0.0
    bspline = kind == GeomAbs_BSplineCurve
    return [
        float(convexity == occ.CONCAVE), float(convexity == occ.CONVEX), float(convexity == occ.SMOOTH),
        length, float(kind == GeomAbs_Circle), float(BRep_Tool.IsClosed(edge)),
        float(kind == GeomAbs_Ellipse), float(bspline and not rational), float(bspline and rational),
        float(kind == GeomAbs_Line),
    ]
