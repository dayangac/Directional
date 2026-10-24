# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

import directional as dr

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"
#a linear function x.A, whose gradient on every face is the tangential part of A
A = np.array([0.3, -1.2, 0.7])


@pytest.fixture(scope="module", params=["spherers.obj", "lilium.obj"])
def mesh(request):
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / request.param))


def tangential(mesh, vectors):
    return vectors-np.sum(vectors*mesh.faceNormals, axis=1)[:, None]*mesh.faceNormals


def assert_sparse_equal(M1, M2, atol=1e-12):
    assert M1.shape==M2.shape
    assert abs(M1-M2).max()<=atol


def test_gradients_of_a_linear_function_are_exact(mesh):
    for G, samples in ((dr.conf_gradient_matrix_2D(mesh), mesh.V), (dr.non_conf_gradient_matrix_2D(mesh), mesh.midEdges)):
        assert G.shape==(3*len(mesh.F), len(samples))
        np.testing.assert_allclose((G@(samples@A)).reshape(-1, 3), tangential(mesh, A), atol=1e-10)


@pytest.mark.parametrize("isIntrinsic", [False, True])
def test_curl_of_gradient_and_divergence_of_rotated_cogradient_vanish(mesh, isIntrinsic):
    rng = np.random.default_rng(0)
    G = dr.conf_gradient_matrix_2D(mesh, isIntrinsic)
    Ge = dr.non_conf_gradient_matrix_2D(mesh, isIntrinsic)
    C = dr.curl_matrix_2D(mesh, isIntrinsic)
    D = dr.div_matrix_2D(mesh, isIntrinsic)
    J = dr.face_vector_rotation_matrix_2D(mesh, isIntrinsic)
    assert G.shape==((2 if isIntrinsic else 3)*len(mesh.F), len(mesh.V))
    assert C.shape==(len(mesh.innerEdges), G.shape[0])
    assert D.shape==(len(mesh.V), G.shape[0])
    np.testing.assert_allclose(C@G@rng.normal(size=len(mesh.V)), 0.0, atol=1e-10)
    #the divergence of a rotated cogradient has a flux term at boundary vertices
    innerVertices = np.setdiff1d(np.arange(len(mesh.V)), np.concatenate([[]]+mesh.boundaryLoops).astype(int))
    np.testing.assert_allclose((D@J@Ge@rng.normal(size=len(mesh.EV)))[innerVertices], 0.0, atol=1e-10)


def test_branched_curl_of_a_matched_gradient_field_vanishes(mesh):
    gradient = (dr.conf_gradient_matrix_2D(mesh)@np.sin(3.0*mesh.V[:, 0])).reshape(-1, 3)
    rawField = dr.CartesianField(dr.PCFaceTangentBundle(mesh), dr.fieldTypeEnum.RAW_FIELD, 2)
    rawField.set_extrinsic_field(np.hstack([gradient, -gradient]))
    dr.curl_matching(rawField)
    C = dr.curl_matrix_2D(mesh, N=2, matching=rawField.matching)
    assert C.shape==(2*len(mesh.innerEdges), 6*len(mesh.F))
    np.testing.assert_allclose(C@rawField.extField.ravel(), 0.0, atol=1e-10)


def test_intrinsic_extrinsic_conversions(mesh):
    EI = dr.face_extrinsic_to_intrinsic_matrix_2D(mesh)
    IE = dr.face_intrinsic_to_extrinsic_matrix_2D(mesh)
    assert_sparse_equal(EI@IE, scipy.sparse.identity(2*len(mesh.F)))
    assert_sparse_equal(IE, EI.T)
    vectors = np.random.default_rng(0).normal(size=(len(mesh.F), 3))
    np.testing.assert_allclose((IE@EI@vectors.ravel()).reshape(-1, 3), tangential(mesh, vectors), atol=1e-12)


def test_rotation_is_a_quarter_turn_in_the_tangent_planes(mesh):
    J = dr.face_vector_rotation_matrix_2D(mesh, isIntrinsic=True)
    assert_sparse_equal(J@J, -scipy.sparse.identity(2*len(mesh.F)))
    vectors = tangential(mesh, np.random.default_rng(0).normal(size=(len(mesh.F), 3)))
    rotated = (dr.face_vector_rotation_matrix_2D(mesh)@vectors.ravel()).reshape(-1, 3)
    np.testing.assert_allclose(rotated, np.cross(mesh.faceNormals, vectors), atol=1e-12)


def test_mass_matrices(mesh):
    for dim in (1, 2, 3):
        np.testing.assert_allclose(dr.face_mass_matrix_2D(mesh, dim=dim).diagonal(), np.repeat(mesh.faceAreas, dim))
        np.testing.assert_allclose(dr.face_mass_matrix_2D(mesh, True, dim).diagonal(), np.repeat(1.0/mesh.faceAreas, dim))
    edgeMasses = dr.edge_diamond_mass_matrix_2D(mesh).diagonal()
    #a third of the area of each adjacent face
    np.testing.assert_allclose(edgeMasses, np.sum(np.where(mesh.EF>=0, mesh.faceAreas[mesh.EF], 0.0), axis=1)/3.0)
    np.testing.assert_allclose(dr.edge_diamond_mass_matrix_2D(mesh, True).diagonal(), 1.0/edgeMasses)


def test_vector_matrices_of_degree_N_act_per_vector(mesh):
    gradient = dr.conf_gradient_matrix_2D(mesh)@(mesh.V@A)
    G2 = dr.conf_gradient_matrix_2D(mesh, N=2)
    assert G2.shape==(6*len(mesh.F), 2*len(mesh.V))
    #two functions, interleaved per vertex
    values = np.column_stack([mesh.V@A, -mesh.V@A]).ravel()
    np.testing.assert_allclose((G2@values).reshape(-1, 2, 3), np.stack([gradient.reshape(-1, 3), -gradient.reshape(-1, 3)], axis=1), atol=1e-10)


def test_fem_matrices_reject_bad_input(mesh):
    with pytest.raises(ValueError):
        dr.conf_gradient_matrix_2D(mesh, N=0)
    with pytest.raises(ValueError):
        dr.curl_matrix_2D(mesh, N=2, matching=np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        dr.face_mass_matrix_2D(mesh, dim=0)


def test_vertex_mass_matrices_integrate_piecewise_linear_functions(mesh):
    M = dr.conf_mass_matrix_2D(mesh)
    lumped = dr.lumped_voronoi_mass_matrix_2D(mesh)
    assert M.shape==lumped.shape==(len(mesh.V), len(mesh.V))
    assert abs(M-M.T).max()==0.0
    #lumping sums the rows, into a third of the area of the adjacent faces
    np.testing.assert_allclose(np.asarray(M.sum(axis=1)).ravel(), lumped.diagonal(), atol=1e-15)
    np.testing.assert_allclose(lumped.diagonal(), np.bincount(mesh.F.ravel(), np.repeat(mesh.faceAreas/3.0, 3), minlength=len(mesh.V)))
    #the square of a linear function is quadratic, so the edge-midpoint rule integrates it exactly
    f = mesh.V@A
    midValues = (f[mesh.F]+f[mesh.F[:, [1, 2, 0]]])/2.0
    assert f@M@f==pytest.approx(np.sum(mesh.faceAreas*np.sum(midValues**2, axis=1)/3.0), rel=1e-12)


def test_vertex_mass_matrices_of_vector_quantities(mesh):
    for massMatrix in (dr.conf_mass_matrix_2D, dr.lumped_voronoi_mass_matrix_2D):
        assert abs(massMatrix(mesh, dim=3)-scipy.sparse.kron(massMatrix(mesh), scipy.sparse.identity(3))).max()==0.0
        with pytest.raises(ValueError):
            massMatrix(mesh, dim=0)
