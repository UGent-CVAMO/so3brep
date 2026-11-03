"""OpenCASCADE helpers for the preprocessing pipeline (pythonocc-core only).

STEP loading and normalisation, face adjacency, UV grids, edge samples with
convexity, tessellation, and a process runner that can kill hung kernel calls.
Adapted from occwl (https://github.com/AutodeskAILab/occwl).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer

CONVEX, CONCAVE, SMOOTH = "convex", "concave", "smooth"
SMOOTH_ANGLE = 0.0872664626  # 5 degrees


# --------------------------------------------------------------------------
# STEP loading and normalisation
# --------------------------------------------------------------------------

def load_step(path) -> "TopoDS_Compound":
    """Read a STEP file as a single compound."""
    shape = read_step_file(str(path), as_compound=True)
    if isinstance(shape, TopoDS_Compound):
        return shape
    compound, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, shape)
    return compound


def pick_body(compound):
    """Return the single solid of a compound, else the compound itself."""
    topo = TopologyExplorer(compound, ignore_orientation=True)
    solids = list(topo.solids())
    if len(solids) == 1:
        return solids[0]
    if not solids and topo.number_of_faces() == 0:
        raise ValueError("No solids or faces found")
    return compound


def load_body(path):
    """Load a STEP file as one body with location layers removed."""
    body = pick_body(load_step(path))
    return BRepBuilderAPI_Copy(body.Located(TopLoc_Location())).Shape()


def scale_to_unit_radius(shape):
    """Move the surface centre of mass to the origin, farthest vertex to radius 1."""
    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    com = props.CentreOfMass()
    cx, cy, cz = com.X(), com.Y(), com.Z()
    max_dist = 0.0
    for vertex in TopologyExplorer(shape, ignore_orientation=True).vertices():
        p = BRep_Tool.Pnt(vertex)
        dx, dy, dz = p.X() - cx, p.Y() - cy, p.Z() - cz
        max_dist = max(max_dist, (dx * dx + dy * dy + dz * dz) ** 0.5)
    if max_dist < 1e-8:
        max_dist = 1.0
    shift, scale = gp_Trsf(), gp_Trsf()
    shift.SetTranslation(gp_Vec(-cx, -cy, -cz))
    scale.SetScale(gp_Pnt(0, 0, 0), 1.0 / max_dist)
    return BRepBuilderAPI_Transform(shape, scale.Multiplied(shift), True).Shape()


def rotation_angles(key: str) -> np.ndarray:
    """Three Euler angles in [0, 2 pi) drawn from a seed that is a hash of ``key``."""
    rng = np.random.RandomState(zlib.crc32(key.encode("utf-8")))
    return rng.uniform(0, 2 * np.pi, size=3)


def rotation_matrix(angles: Sequence[float]) -> np.ndarray:
    """Matrix R = Rx Ry Rz of the transform applied by :func:`rotate`."""
    (c0, c1, c2), (s0, s1, s2) = np.cos(angles), np.sin(angles)
    rx = np.array([[1, 0, 0], [0, c0, -s0], [0, s0, c0]])
    ry = np.array([[c1, 0, s1], [0, 1, 0], [-s1, 0, c1]])
    rz = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
    return rx @ ry @ rz


def rotate(shape, angles: Sequence[float]):
    """Rotate a shape about the origin by Rx(a0) Ry(a1) Rz(a2) (geometry is copied)."""
    origin = gp_Pnt(0, 0, 0)
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(origin, gp_Dir(1, 0, 0)), float(angles[0]))
    for axis, angle in (((0, 1, 0), angles[1]), ((0, 0, 1), angles[2])):
        step = gp_Trsf()
        step.SetRotation(gp_Ax1(origin, gp_Dir(*axis)), float(angle))
        trsf = trsf.Multiplied(step)
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


@contextlib.contextmanager
def _silence_c_output():
    """Redirect C-level stdout and stderr to /dev/null."""
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def write_step(shape, path: Path) -> None:
    """Write a shape to a STEP file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _silence_c_output():
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(str(path))


# --------------------------------------------------------------------------
# Topology: face adjacency
# --------------------------------------------------------------------------

def _is_left_of(face, edge) -> bool:
    """True if ``face`` lies to the left of the oriented ``edge``."""
    found = False
    for wire in TopologyExplorer(face, ignore_orientation=True).wires():
        for wire_edge in WireExplorer(wire).ordered_edges():
            if hash(wire_edge) == hash(edge):
                if (wire_edge.Orientation() == TopAbs_REVERSED) == (edge.Orientation() == TopAbs_REVERSED):
                    return True
                found = True
    assert found, "Edge doesn't belong to face"
    return False


def left_and_right_faces(edge, faces: list):
    """Faces left and right of an edge seen from outside, or (None, None)."""
    face1 = faces[0]
    face2 = faces[1] if len(faces) > 1 else faces[0]
    if _is_left_of(face1, edge):
        if hash(face1) != hash(face2) and _is_left_of(face2, edge):
            return None, None
        return face1, face2
    if not _is_left_of(face2, edge):
        return None, None
    return face2, face1


def has_curve(edge) -> bool:
    """False for edges without a 3D curve (e.g. sphere poles)."""
    return BRepAdaptor_Curve(edge).Is3DCurve()


def face_adjacency(shape) -> Tuple[List, dict]:
    """Faces of a shape and its oriented face adjacency.

    Returns ``(faces, edges)``: the faces in explorer order (the index is the
    node id) and a dict ``(left, right) -> oriented edge``. Both directions of
    an adjacency are present; if two edges join the same face pair the last
    one wins. Non-manifold edges raise, edges whose orientation cannot be
    resolved are skipped.
    """
    topo = TopologyExplorer(shape, ignore_orientation=True)
    faces = list(topo.faces())
    face_index = {hash(face): i for i, face in enumerate(faces)}
    oriented = {
        (hash(edge), edge.Orientation() == TopAbs_REVERSED)
        for wire in topo.wires()
        for edge in WireExplorer(wire).ordered_edges()
    }

    def known(edge) -> bool:
        return (hash(edge), edge.Orientation() == TopAbs_REVERSED) in oriented

    adjacency = {}
    for edge in topo.edges():
        if not has_curve(edge):
            continue
        adjacent = list(topo.faces_from_edge(edge))
        if len(adjacent) < 2:
            continue
        if len(adjacent) > 2:
            raise RuntimeError("Expected a manifold, an edge must be incident on one/two faces")
        try:
            left, right = left_and_right_faces(edge, adjacent)
        except Exception:
            continue
        if left is None or right is None:
            continue
        reverse = edge.Reversed()
        if not (known(edge) and known(reverse)):
            continue
        left_id, right_id = face_index[hash(left)], face_index[hash(right)]
        adjacency[(left_id, right_id)] = edge
        adjacency[(right_id, left_id)] = reverse
    return faces, adjacency


# --------------------------------------------------------------------------
# Faces: classifier and UV grid
# --------------------------------------------------------------------------

def strip_location(face):
    """The face with its location layer removed."""
    return face.Located(TopLoc_Location())


def trim_classifier(face) -> BRepTopAdaptor_FClass2d:
    """Classifier telling which UV points lie inside the trimmed face."""
    return BRepTopAdaptor_FClass2d(face, 1e-9)


def uv_bounds(face) -> Tuple[float, float, float, float]:
    """UV box (umin, umax, vmin, vmax) of a face with ordered limits."""
    u0, u1, v0, v1 = breptools.UVBounds(face)
    return min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1)


def uv_grid(face, classifier, num_u: int, num_v: int):
    """Sample a face on a ``num_u x num_v`` UV grid.

    Returns float32 ``points`` and ``normals`` (U, V, 3) and a bool ``mask``
    (U, V) true where the sample is inside or on the trimmed face. The grid is
    flipped along u for reversed faces and the normals follow the face orientation.
    """
    u_lo, u_hi, v_lo, v_hi = uv_bounds(face)
    adaptor = BRepAdaptor_Surface(face)
    reverse = face.Orientation() == TopAbs_REVERSED
    n = num_u * num_v
    points = np.empty((n, 3), dtype=np.float32)
    normals = np.empty((n, 3), dtype=np.float32)
    mask = np.empty(n, dtype=bool)
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    k = 0
    for i in range(num_u):
        u = (1.0 - float(i) / (num_u - 1)) * u_lo + float(i) / (num_u - 1) * u_hi
        for j in range(num_v):
            v = (1.0 - float(j) / (num_v - 1)) * v_lo + float(j) / (num_v - 1) * v_hi
            adaptor.D1(u, v, point, du, dv)
            points[k] = (point.X(), point.Y(), point.Z())
            nx = du.Y() * dv.Z() - du.Z() * dv.Y()
            ny = du.Z() * dv.X() - du.X() * dv.Z()
            nz = du.X() * dv.Y() - du.Y() * dv.X()
            norm = (nx * nx + ny * ny + nz * nz) ** 0.5
            if norm > 1e-12:
                nx, ny, nz = nx / norm, ny / norm, nz / norm
            normals[k] = (-nx, -ny, -nz) if reverse else (nx, ny, nz)
            state = int(classifier.Perform(gp_Pnt2d(u, v)))
            mask[k] = state in (0, 2)
            k += 1
    points, normals = points.reshape(num_u, num_v, 3), normals.reshape(num_u, num_v, 3)
    mask = mask.reshape(num_u, num_v)
    if reverse:
        points, normals, mask = points[::-1], normals[::-1], mask[::-1]
    return points, normals, mask


def face_normal(face, uv) -> np.ndarray:
    """Unit surface normal at ``uv``, flipped for reversed faces (zeros if undefined)."""
    surface = BRep_Tool.Surface(face)
    props = GeomLProp_SLProps(surface, uv[0], uv[1], 1, 1e-9)
    if not props.IsNormalDefined():
        return np.zeros(3)
    d = props.Normal()
    normal = np.array([d.X(), d.Y(), d.Z()])
    return -normal if face.Orientation() == TopAbs_REVERSED else normal


# --------------------------------------------------------------------------
# Edges: samples and convexity
# --------------------------------------------------------------------------

def _curve_interval(edge) -> Tuple[float, float]:
    """Parameter interval (a, b) of the edge curve, ordered."""
    _, first, last = BRep_Tool.Curve(edge)
    return (first, last) if first < last else (last, first)


def _edge_point(curve, u: float) -> np.ndarray:
    p = curve.Value(u)
    return np.array([p.X(), p.Y(), p.Z()])


def _edge_tangent(edge, curve, u: float) -> np.ndarray:
    point, d = gp_Pnt(), gp_Vec()
    curve.D1(u, point, d)
    d.Normalize()
    tangent = np.array([d.X(), d.Y(), d.Z()])
    return -tangent if edge.Orientation() == TopAbs_REVERSED else tangent


def _uniform_params(edge, n: int) -> List[float]:
    a, b = _curve_interval(edge)
    params = [(1.0 - i / (n - 1)) * a + i / (n - 1) * b for i in range(n)]
    return params[::-1] if edge.Orientation() == TopAbs_REVERSED else params


def _arclength_params(edge, curve, n: int, chords: int = 30) -> List[float]:
    """Parameters spaced at equal arc length (chord approximation), edge-oriented."""
    a, b = _curve_interval(edge)
    us = [(1.0 - i / (chords - 1)) * a + i / (chords - 1) * b for i in range(chords)]
    points = [_edge_point(curve, u) for u in us]
    total = 0.0
    lengths = []
    for p, q in zip(points[:-1], points[1:]):
        lengths.append(np.linalg.norm(q - p))
        total += lengths[-1]
    fraction, cumulative = [0.0], 0.0
    for length in lengths:
        cumulative += length
        fraction.append(cumulative / total)
    params, k = [], 0
    for i in range(n):
        want = i / (n - 1)
        while fraction[k] < want:
            k += 1
            if k >= len(fraction) - 1:
                break
        u_low, f_low = (us[0], fraction[0]) if k == 0 else (us[k - 1], fraction[k - 1])
        u_high, f_high = us[k], fraction[k]
        assert f_low <= want <= f_high
        if f_high - f_low <= 0.0:
            params.append(u_low)
        else:
            lo, hi = (u_low, u_high) if u_low < u_high else (u_high, u_low)
            t = (want - f_low) / (f_high - f_low)
            params.append((1.0 - t) * lo + t * hi)
    return params[::-1] if edge.Orientation() == TopAbs_REVERSED else params


class EdgeSamples:
    """Points, tangents and both adjacent face normals along an oriented edge.

    ``good`` is False when no left and right face could be assigned. Arrays
    are (n, 3) and follow the edge direction.
    """

    def __init__(self, edge, faces: list, n: int, arclength: bool):
        self.good = True
        left, right = left_and_right_faces(edge, faces)
        if left is None or right is None:
            self.good = False
            return
        curve = BRep_Tool.Curve(edge)[0]
        params = _arclength_params(edge, curve, n) if arclength else _uniform_params(edge, n)
        self.points = np.stack([_edge_point(curve, u) for u in params])
        self.tangents = np.stack([_edge_tangent(edge, curve, u) for u in params])
        self.left_normals = self._normals(edge, left, params)
        self.right_normals = self._normals(edge, right, params)

    @staticmethod
    def _normals(edge, face, params) -> np.ndarray:
        pcurve = BRepAdaptor_Curve2d(edge, face)
        normals = []
        for u in params:
            uv = gp_Pnt2d()
            pcurve.D0(u, uv)
            normals.append(face_normal(face, np.array([uv.X(), uv.Y()])))
        return np.stack(normals)

    def convexity(self, angle_tol: float = SMOOTH_ANGLE) -> str:
        """CONVEX, CONCAVE or SMOOTH from the normals along the edge."""
        dots = np.multiply(self.left_normals, self.right_normals).sum(1)
        if dots.mean() > np.cos(angle_tol):
            return SMOOTH
        cross = np.cross(self.left_normals, self.right_normals, axis=1)
        return CONVEX if np.multiply(cross, self.tangents).sum(1).sum() > 0.0 else CONCAVE


# --------------------------------------------------------------------------
# Tessellation
# --------------------------------------------------------------------------

def tessellate(shape, linear: float, angular: float):
    """Triangulate a shape.

    Returns ``(vertices, triangles, face_ids)``: float64 (N, 3), int32 (M, 3)
    with the winding following face orientation, and int32 (M,) index of the
    source face in explorer order.
    """
    mesh = BRepMesh_IncrementalMesh(shape, linear, False, angular, False)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Tessellation failed")
    vertices, triangles, face_ids = [], [], []
    face_index = 0
    offset = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            trsf = location.Transformation()
            for i in range(1, triangulation.NbNodes() + 1):
                p = triangulation.Node(i)
                p.Transform(trsf)
                vertices.append([p.X(), p.Y(), p.Z()])
            reverse = face.Orientation() == TopAbs_REVERSED
            for i in range(1, triangulation.NbTriangles() + 1):
                n1, n2, n3 = triangulation.Triangle(i).Get()
                base = offset - 1
                triangles.append([n1 + base, n3 + base, n2 + base] if reverse else [n1 + base, n2 + base, n3 + base])
                face_ids.append(face_index)
            offset += triangulation.NbNodes()
        face_index += 1
        explorer.Next()
    return (
        np.array(vertices, dtype=np.float64),
        np.array(triangles, dtype=np.int32),
        np.array(face_ids, dtype=np.int32),
    )


# --------------------------------------------------------------------------
# Process runner
# --------------------------------------------------------------------------

def _child(fn: Callable, arg, conn) -> None:
    try:
        conn.send(fn(arg))
    except BaseException as error:
        conn.send((False, f"{type(error).__name__}: {error}"))
    finally:
        conn.close()


def run_isolated(fn: Callable, args: Iterable, workers: int, timeout: float) -> Iterator[Tuple[object, bool, str]]:
    """Run ``fn(arg)`` for each arg in its own forked process.

    ``fn`` returns ``(ok, message)``. A call that exceeds ``timeout`` seconds is
    killed (a hung kernel call cannot be interrupted from inside), as is one
    that crashes the interpreter. Yields ``(arg, ok, message)`` as calls finish.
    """
    ctx = mp.get_context("fork")
    pending = iter(args)
    running = {}
    exhausted = False
    while running or not exhausted:
        while not exhausted and len(running) < workers:
            try:
                arg = next(pending)
            except StopIteration:
                exhausted = True
                break
            parent, child = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_child, args=(fn, arg, child), daemon=True)
            proc.start()
            child.close()
            running[proc.pid] = (proc, parent, arg, time.monotonic())
        for pid in list(running):
            proc, conn, arg, started = running[pid]
            result = None
            if conn.poll():
                try:
                    result = conn.recv()
                except EOFError:
                    result = (False, f"worker died (exit code {proc.exitcode})")
            elif not proc.is_alive():
                result = (False, f"worker died (exit code {proc.exitcode})")
            elif time.monotonic() - started > timeout:
                proc.kill()
                result = (False, f"timed out after {timeout:.0f}s")
            if result is not None:
                proc.join()
                conn.close()
                del running[pid]
                yield (arg, *result)
        time.sleep(0.005)
