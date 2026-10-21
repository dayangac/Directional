# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np
import pytest

import directional as dr

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"
#torus.obj has its axis along y, center radius R and tube radius r
R, r = 1.0, 0.25


@pytest.fixture(scope="module")
def torus():
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / "torus.obj"))


def test_principal_directions_form_orthonormal_frame_with_normals(torus):
    minDirections, maxDirections = torus.minVertexPrincipalDirections, torus.maxVertexPrincipalDirections
    for directions in (minDirections, maxDirections):
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-10)
        np.testing.assert_allclose(np.sum(directions*torus.vertexNormals, axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(np.sum(minDirections*maxDirections, axis=1), 0, atol=1e-10)


def test_principal_curvatures_are_ordered(torus):
    assert torus.vertexPrincipalCurvatures.shape==(len(torus.V), 2)
    assert np.all(torus.vertexPrincipalCurvatures[:, 0]<=torus.vertexPrincipalCurvatures[:, 1])


def test_principal_curvatures_match_analytic_torus(torus):
    rho = np.linalg.norm(torus.V[:, [0, 2]], axis=1)
    cosTheta = (rho-R)/r
    #tolerances cover the discretization error of the 576-vertex mesh (measured 0.19 and 12%)
    np.testing.assert_allclose(torus.vertexPrincipalCurvatures[:, 0], cosTheta/rho, atol=0.25)
    np.testing.assert_allclose(torus.vertexPrincipalCurvatures[:, 1], 1/r, rtol=0.15)


def test_readDMAT():
    constFaces = dr.readDMAT(str(TUTORIAL_DATA_PATH / "spheres_constFaces.dmat"))
    assert constFaces.shape==(98, 1)
    assert constFaces[0, 0]==9
    with pytest.raises(RuntimeError):
        dr.readDMAT(str(TUTORIAL_DATA_PATH / "missing.dmat"))


@pytest.mark.parametrize("fileName, genus", [("spherers.obj", 0), ("torus.obj", 1), ("fertility.obj", 4)])
def test_closed_mesh_topology(fileName, genus):
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / fileName))
    assert mesh.numGenerators==genus
    assert mesh.boundaryLoops==[]
    np.testing.assert_array_equal(mesh.innerEdges, np.arange(len(mesh.EV)))


def test_boundary_loops_are_closed_boundary_vertex_chains():
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    boundaryEdges = np.flatnonzero((mesh.EF<0).any(axis=1))
    assert len(mesh.boundaryLoops)==1
    loop = np.array(mesh.boundaryLoops[0])
    assert len(loop)==len(boundaryEdges)
    loopEdges = {frozenset(e) for e in zip(loop, np.roll(loop, -1))}
    assert loopEdges=={frozenset(e) for e in mesh.EV[boundaryEdges]}
    np.testing.assert_array_equal(np.sort(np.concatenate([mesh.innerEdges, boundaryEdges])), np.arange(len(mesh.EV)))


def test_mid_edges_and_face_edge_signs():
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    np.testing.assert_allclose(mesh.midEdges, (mesh.V[mesh.EV[:, 0]]+mesh.V[mesh.EV[:, 1]])/2.0)
    #FEs orients each edge along its face, so the signed edge vectors of a face close up
    edgeVectors = mesh.V[mesh.EV[:, 1]]-mesh.V[mesh.EV[:, 0]]
    np.testing.assert_allclose(np.einsum("fi,fix->fx", mesh.FEs, edgeVectors[mesh.FE]), 0.0, atol=1e-12)
    assert set(np.unique(mesh.FEs))=={-1.0, 1.0}
