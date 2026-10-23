# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

import directional as dr

if not hasattr(dr, "mesher"):
    pytest.skip("the module was built without GMP, so without meshing", allow_module_level=True)

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"


@pytest.fixture(scope="module")
def vase():
    return dr.readOFF(str(TUTORIAL_DATA_PATH / "vase.off"))


def integrated(mesh, N):
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / ("vase-" + str(N) + ".rawfield")), dr.PCFaceTangentBundle(mesh))
    dr.principal_matching(rawField)
    intData = dr.IntegrationData(N)
    meshCut, combedField = dr.setup_integration(rawField, intData)
    intData.integralSeamless, intData.roundSeams = True, False
    dr.integrate(combedField, intData, meshCut)
    return intData, meshCut


@pytest.fixture(scope="module", params=[4, 7])
def polyMesh(request, vase):
    intData, meshCut = integrated(vase, request.param)
    mData = dr.setup_mesher(meshCut, intData)
    return (request.param, mData)+dr.mesher(vase, mData)


def halfedges(D, F):
    return np.concatenate([np.column_stack([F[f, :D[f]], np.roll(F[f, :D[f]], -1)]) for f in range(len(D))])


def test_setup_mesher_reduces_the_sign_symmetry(polyMesh):
    N, mData = polyMesh[:2]
    assert mData.N==(N//2 if N%2==0 else N)


def test_faces_are_valid_polygons(polyMesh):
    _, _, V, D, F = polyMesh
    assert np.all(D>=3) and F.shape==(len(D), D.max())
    padding = np.arange(F.shape[1])[None, :]>=D[:, None]
    np.testing.assert_array_equal(F[padding], -1)
    assert F[~padding].min()>=0 and F[~padding].max()<len(V)
    np.testing.assert_array_equal(np.unique(F[~padding]), np.arange(len(V)))  #no unreferenced vertices


def test_polygonal_mesh_is_an_oriented_manifold_with_the_same_topology(vase, polyMesh):
    _, _, V, D, F = polyMesh
    H = halfedges(D, F)
    assert len(np.unique(H, axis=0))==len(H)  #consistently oriented: no halfedge appears twice
    edges = np.unique(np.sort(H, axis=1), axis=0)
    assert len(V)-len(edges)+len(D)==vase.eulerChar


def test_vertices_lie_on_the_surface(vase, polyMesh):
    _, _, V, _, _ = polyMesh
    _, nearFaces = cKDTree(vase.barycenters).query(V, k=12)
    a, b, c = (vase.V[vase.F[nearFaces, k]] for k in range(3))
    #barycentric coordinates in the plane of each nearby face
    n = np.cross(b-a, c-a)
    coords = np.stack([np.sum(np.cross(c-b, V[:, None]-b)*n, axis=2), np.sum(np.cross(a-c, V[:, None]-c)*n, axis=2),
                       np.sum(np.cross(b-a, V[:, None]-a)*n, axis=2)], axis=2)/np.sum(n*n, axis=2)[:, :, None]
    planeDistances = np.abs(np.sum((V[:, None]-a)*n, axis=2))/np.linalg.norm(n, axis=2)
    onFace = (coords.min(axis=2)>-1e-8) & (planeDistances<1e-8*vase.avgEdgeLength)
    assert np.all(onFace.any(axis=1))


def test_quad_mesh_of_a_4_field(polyMesh):
    N, _, _, D, _ = polyMesh
    if N==4:
        assert np.mean(D==4)>0.95


def test_meshing_rejects_bad_input(vase):
    intData, meshCut = integrated(vase, 4)
    with pytest.raises(ValueError):
        dr.setup_mesher(meshCut, dr.IntegrationData(4))
    otherMesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    with pytest.raises(ValueError):
        dr.mesher(otherMesh, dr.setup_mesher(meshCut, intData))
