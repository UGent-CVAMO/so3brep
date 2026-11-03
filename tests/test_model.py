"""SO3brep on random face graphs: every variant runs, node irreps are equivariant, outputs are invariant."""

import pytest
import torch
from e3nn.o3 import spherical_harmonics
from torch_geometric.data import Batch, Data

from dataloaders.rotate import haar, rotate_pyg
from eval.equivariance import irreps_of_degree
from models.so3brep import SO3brep, wigner_d

torch.manual_seed(0)


def random_part(n_faces: int = 7) -> Data:
    """A ring of faces with random geometry and the fields SO3brep reads."""
    centre = torch.randn(n_faces, 3)
    normal = torch.nn.functional.normalize(torch.randn(n_faces, 3), dim=-1)
    u = torch.nn.functional.normalize(torch.cross(normal, torch.randn(n_faces, 3), dim=-1), dim=-1)
    src = torch.arange(n_faces)
    edge_index = torch.cat([torch.stack([src, (src + 1) % n_faces]), torch.stack([(src + 1) % n_faces, src])], 1)
    n_edges = edge_index.shape[1]
    curve = centre[edge_index[0]].unsqueeze(1) + 0.1 * torch.randn(n_edges, 10, 3)
    return Data(
        center_nuv=torch.cat([centre, normal, u, torch.cross(normal, u, dim=-1)], dim=-1), pos=centre,
        x_grid_local=torch.randn(n_faces, 7, 10, 10), vision_grids=torch.rand(n_faces, 6, 6, 12),
        face_feat=torch.rand(n_faces, 7), edge_index=edge_index, edge_attr=torch.rand(n_edges, 10),
        edge_curve_pts=torch.cat([curve, torch.randn(n_edges, 10, 9)], dim=-1), edge_mid=curve[:, 4],
        num_nodes=n_faces,
    )


VARIANTS = [{}, {"message": "attention"}, {"message": "concat"}, {"message": "invariant"}, {"l_max": 1},
            {"l_max": 0}, {"rays": False}, {"uv_grid": False}, {"surface_attrs": False}, {"attention": False},
            {"use_edge_features": False}, {"use_node_features": False}]


@pytest.mark.parametrize("options", VARIANTS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "paper")
def test_variants_run(options):
    parts = Batch.from_data_list([random_part(5 + i) for i in range(3)])
    assert SO3brep(8, True, **options).eval()(parts).shape == (parts.num_nodes, 8)
    assert SO3brep(8, False, **options).eval()(parts).shape == (3, 8)


def test_lifting_matches_world_frame_harmonics():
    """Harmonics computed in a local frame and lifted as in SO3brep.node_features equal those of the
    same directions in the world frame. The equivariance test cannot catch a wrong basis here."""
    frames = haar(16).transpose(1, 2)  # rows u, v, n, as returned by local_frames
    d_local = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    d_world = torch.einsum("nij,ni->nj", frames, d_local)
    y = {l: spherical_harmonics([l], d_local, True, normalization="component")[:, None] for l in (1, 2)}
    lifted1 = torch.bmm(y[1], frames)[:, 0]
    lifted2 = torch.bmm(y[2], wigner_d(2, frames.transpose(1, 2)).transpose(1, 2))[:, 0]
    assert torch.allclose(lifted1, spherical_harmonics([1], d_world, True, normalization="component"), atol=1e-5)
    assert torch.allclose(lifted2, spherical_harmonics([2], d_world, True, normalization="component"), atol=1e-4)


def test_irreps_are_equivariant():
    net = SO3brep(8, True).eval()
    part = Batch.from_data_list([random_part()])
    cols, irreps = irreps_of_degree(net.gnn.irreps_node, {0, 1, 2})
    with torch.no_grad():
        h = net.encode(part)[2][:, cols]
        for R in haar(3):
            h_rot = net.encode(rotate_pyg(part, R))[2][:, cols]
            assert (h_rot - h @ irreps.D_from_matrix(R).T).norm() / h_rot.norm() < 1e-4


@pytest.mark.parametrize("message,invariant", [("cg", True), ("invariant", True), ("concat", False)])
def test_output_invariance(message, invariant):
    net = SO3brep(8, True, message=message).eval()
    part = random_part()
    with torch.no_grad():
        ref = net(Batch.from_data_list([part]))
        for R in haar(3):
            out = net(Batch.from_data_list([rotate_pyg(part, R)]))
            assert torch.allclose(out, ref, atol=1e-4) == invariant
