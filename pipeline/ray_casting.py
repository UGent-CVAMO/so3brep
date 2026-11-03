"""Hemisphere ray casting from face centres onto the tessellated part."""

from __future__ import annotations

from typing import List

import numpy as np
import trimesh

from pipeline import occ

LINEAR_DEFLECTION = 0.03
ANGULAR_DEFLECTION = 0.15


class MeshRayCaster:
    """Closest-hit ray casting against a triangle mesh with a face id per triangle."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, face_ids: np.ndarray,
                 linear_deflection: float = LINEAR_DEFLECTION):
        self.vertices, self.triangles, self.face_ids = vertices, triangles, face_ids
        self.linear_deflection = linear_deflection
        # process=False keeps the tessellation order, so face ids stay valid.
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False, validate=False)
        try:
            self.intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        except Exception:
            self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    @classmethod
    def from_shape(cls, shape) -> "MeshRayCaster":
        """Tessellate a shape and build the caster."""
        vertices, triangles, face_ids = occ.tessellate(shape, LINEAR_DEFLECTION, ANGULAR_DEFLECTION)
        return cls(vertices, triangles, face_ids)

    def intersect(self, origins: np.ndarray, directions: np.ndarray):
        """Closest hit per ray: distances (inf if none), face ids (-1) and triangle normals."""
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.divide(directions, norms, where=norms > 0, out=np.zeros_like(directions))
        n = len(origins)
        dists = np.full(n, np.inf)
        fids = np.full(n, -1, dtype=np.int32)
        normals = np.zeros((n, 3))
        locations, ray_index, tri_index = self.intersector.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False,
        )
        if len(locations) > 0:
            d = np.linalg.norm(locations - origins[ray_index], axis=1)
            order = np.lexsort((d, ray_index))
            _, first = np.unique(ray_index[order], return_index=True)
            pick = order[first]
            rays = ray_index[pick]
            dists[rays] = d[pick]
            fids[rays] = self.face_ids[tri_index[pick]]
            normals[rays] = self.mesh.face_normals[tri_index[pick]]
        return dists, fids, normals


def fibonacci_hemisphere(n: int) -> np.ndarray:
    """``n`` unit directions on the hemisphere z > 0 (Fibonacci lattice), in local coordinates."""
    idx = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = 1 - (idx + 0.5) / n
    r = np.sqrt(1 - z * z)
    t = phi * idx
    x, y, zz = np.eye(3)
    dirs = r[:, None] * np.cos(t)[:, None] * x + r[:, None] * np.sin(t)[:, None] * y + z[:, None] * zz
    norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.divide(dirs, norm, out=np.zeros_like(dirs), where=norm != 0)


def cast_from_faces(caster: MeshRayCaster, centers: np.ndarray, axes: List[np.ndarray],
                    n_elev: int, n_azim: int, max_dist: float) -> dict:
    """Cast the same direction set along and against each face normal.

    Directions are the lattice rotated into each face frame (forward) and
    reflected in the tangent plane (backward). Returns arrays with a leading
    face axis: ``occ``, ``dist``, ``dot`` and ``hit`` (face id), each for
    ``fwd`` and ``back`` as (F, n_elev, n_azim).
    """
    per_face = n_elev * n_azim
    n_faces = len(centers)
    epsilon = max(2.0 * caster.linear_deflection, 1e-3)
    axes_arr = np.asarray(axes, dtype=np.float64)
    fwd = np.einsum("rj,fij->fri", fibonacci_hemisphere(per_face), axes_arr)
    normals = axes_arr[:, :, 2]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
    back = fwd - 2.0 * np.einsum("fri,fi->fr", fwd, normals)[:, :, None] * normals[:, None, :]

    paired = np.empty((n_faces, 2, per_face, 3))
    paired[:, 0], paired[:, 1] = fwd, back
    origins = np.asarray(centers, dtype=np.float64)[:, None, None, :] + epsilon * paired
    dists, fids, hit_normals = caster.intersect(origins.reshape(-1, 3), paired.reshape(-1, 3))
    hit = dists < np.inf
    dists[hit] += epsilon
    dists[hit & (dists > max_dist)] = np.inf

    shape = (n_faces, 2, n_elev, n_azim)
    occupied = ~np.isinf(dists)
    dot = np.zeros(len(dists))
    dot[occupied] = np.einsum("ij,ij->i", paired.reshape(-1, 3)[occupied], hit_normals[occupied])
    out = {
        "occ": occupied.astype(np.float64).reshape(shape),
        "dist": np.where(occupied, dists, 0.0).reshape(shape),
        "dot": dot.reshape(shape),
        "hit": fids.reshape(shape),
    }
    return out
