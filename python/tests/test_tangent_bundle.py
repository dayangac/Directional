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
VERTEX_BUNDLES = [dr.IntrinsicVertexTangentBundle, dr.ExtrinsicVertexTangentBundle]
BUNDLES = [dr.PCFaceTangentBundle]+VERTEX_BUNDLES


#lilium has a boundary
@pytest.fixture(scope="module")
def mesh():
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))


def test_bundle_types(mesh):
    for TangentBundle in BUNDLES:
        tb = TangentBundle(mesh)
        assert tb.discTangType()==(dr.discTangTypeEnum.FACE_SPACES if TangentBundle is dr.PCFaceTangentBundle else dr.discTangTypeEnum.VERTEX_SPACES)
        assert tb.hasEmbedding() and tb.intDimension==2
        assert tb.hasCochainSequence()==(TangentBundle is dr.PCFaceTangentBundle)
        assert tb.mesh is mesh


def test_face_bundle_combinatorics(mesh):
    tb = dr.PCFaceTangentBundle(mesh)
    np.testing.assert_array_equal(tb.oneRing, mesh.FE)
    np.testing.assert_array_equal(tb.innerAdjacencies, mesh.innerEdges)
    np.testing.assert_array_equal(tb.cycleNormals, mesh.vertexNormals)


@pytest.mark.parametrize("TangentBundle", VERTEX_BUNDLES)
def test_vertex_bundle_combinatorics(mesh, TangentBundle):
    tb = TangentBundle(mesh)
    np.testing.assert_array_equal(tb.oneRing, mesh.VE)
    #all edges, including the boundary ones (a TODO in the library)
    np.testing.assert_array_equal(tb.innerAdjacencies, np.arange(len(mesh.EV)))
    np.testing.assert_array_equal(tb.cycleNormals, mesh.faceNormals)


def test_face_connection_transports_edge_directions(mesh):
    tb = dr.PCFaceTangentBundle(mesh)
    inner = mesh.innerEdges
    np.testing.assert_allclose(np.abs(tb.connection[inner]), 1.0, atol=1e-12)
    np.testing.assert_array_equal(np.delete(tb.connection, inner), 0.0)
    #the shared edge, in the complex basis of each of its faces, is related by the connection
    edges = mesh.V[mesh.EV[inner, 1]]-mesh.V[mesh.EV[inner, 0]]
    inFace = [np.sum(edges*mesh.FBx[mesh.EF[inner, k]], axis=1)+1j*np.sum(edges*mesh.FBy[mesh.EF[inner, k]], axis=1) for k in (0, 1)]
    np.testing.assert_allclose(inFace[0]*tb.connection[inner], inFace[1], atol=1e-12)


@pytest.mark.parametrize("TangentBundle", VERTEX_BUNDLES)
def test_vertex_connection_is_a_rotation(mesh, TangentBundle):
    np.testing.assert_allclose(np.abs(TangentBundle(mesh).connection), 1.0, atol=1e-12)


def test_face_bundle_masses(mesh):
    tb = dr.PCFaceTangentBundle(mesh)
    np.testing.assert_allclose(tb.tangentSpaceMass.diagonal(), mesh.faceAreas)
    np.testing.assert_allclose(tb.invTangentSpaceMass.diagonal(), 1.0/mesh.faceAreas)
    #the harmonic weights of [Brandt et al. 2020], per inner edge
    inner = mesh.innerEdges
    lengths = np.linalg.norm(mesh.V[mesh.EV[inner, 1]]-mesh.V[mesh.EV[inner, 0]], axis=1)
    np.testing.assert_allclose(tb.connectionMass.diagonal(), 3.0*lengths**2/mesh.faceAreas[mesh.EF[inner]].sum(axis=1))


@pytest.mark.parametrize("TangentBundle", VERTEX_BUNDLES)
def test_vertex_bundle_masses_are_voronoi_areas_and_cotangent_weights(mesh, TangentBundle):
    #each edge gets the cotangents of the two angles opposite to it, as in the circumcentric Hodge star
    tb = TangentBundle(mesh)
    np.testing.assert_allclose(tb.tangentSpaceMass.diagonal(), np.bincount(mesh.F.ravel(), np.repeat(mesh.faceAreas/3.0, 3), minlength=len(mesh.V)))
    assert tb.connectionMass.shape==(len(mesh.EV), len(mesh.EV))
    np.testing.assert_allclose(tb.connectionMass.diagonal(), dr.hodge_star_1_matrix(mesh, True)[0].diagonal(), atol=1e-12)


@pytest.mark.parametrize("TangentBundle", BUNDLES)
def test_projections_between_representations(mesh, TangentBundle):
    tb = TangentBundle(mesh)
    rng = np.random.default_rng(0)
    spaces = rng.integers(0, tb.numSpaces, 20)
    vectors = rng.normal(size=(20, 6))
    intrinsic = tb.project_to_intrinsic(spaces, vectors)
    assert intrinsic.shape==(20, 4)
    normals = np.tile(tb.normals[spaces], 2)
    tangential = vectors-np.repeat(np.add.reduceat(vectors*normals, [0, 3], axis=1), 3, axis=1)*normals
    np.testing.assert_allclose(tb.project_to_extrinsic(spaces, intrinsic), tangential, atol=1e-12)
    #no tangent spaces means all of them
    allIntrinsic = rng.normal(size=(tb.numSpaces, 2))
    np.testing.assert_allclose(tb.project_to_extrinsic(np.empty(0, dtype=int), allIntrinsic),
                               tb.project_to_extrinsic(np.arange(tb.numSpaces), allIntrinsic))


def test_face_bundle_interpolation_is_constant_per_face(mesh):
    rawField = dr.power_to_raw(dr.power_field(dr.PCFaceTangentBundle(mesh), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 2), 2)
    faces = np.array([0, 3, 3, 500])
    baryCoords = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [1/3, 1/3, 1/3], [0.0, 0.5, 0.5]])
    sources, normals, field = rawField.tb.interpolate(faces, baryCoords, rawField.intField)
    np.testing.assert_allclose(sources, np.einsum("sk,skx->sx", baryCoords, mesh.V[mesh.F[faces]]))
    np.testing.assert_allclose(normals, mesh.faceNormals[faces])
    np.testing.assert_allclose(field, rawField.extField[faces], atol=1e-12)


@pytest.mark.parametrize("TangentBundle", VERTEX_BUNDLES)
def test_vertex_bundle_interpolation_is_tangent_and_constant_per_face(mesh, TangentBundle):
    #the library's interpolation is still the polyvector average at the face center
    rawField = dr.power_to_raw(dr.power_field(TangentBundle(mesh), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 1), 1)
    faces = np.array([0, 0, 0, 5])
    baryCoords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1/3, 1/3, 1/3], [0.2, 0.3, 0.5]])
    _, normals, field = rawField.tb.interpolate(faces, baryCoords, rawField.intField)
    np.testing.assert_allclose(np.sum(field*normals, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(field[:3], np.tile(field[0], (3, 1)), atol=1e-12)


def test_bundle_methods_reject_bad_input(mesh):
    tb = dr.PCFaceTangentBundle(mesh)
    with pytest.raises(ValueError):
        tb.project_to_intrinsic(np.array([0, 1]), np.ones((1, 3)))
    with pytest.raises(ValueError):
        tb.project_to_intrinsic(np.array([tb.numSpaces]), np.ones((1, 3)))
    with pytest.raises(ValueError):
        tb.project_to_extrinsic(np.array([0]), np.ones((1, 3)))
    with pytest.raises(ValueError):
        tb.project_to_extrinsic(np.empty(0, dtype=int), np.ones((5, 2)))
    with pytest.raises(ValueError):
        tb.interpolate(np.array([0]), np.ones((1, 3)), np.ones((5, 2)))
    with pytest.raises(ValueError):
        tb.interpolate(np.array([len(mesh.F)]), np.ones((1, 3)), np.ones((tb.numSpaces, 2)))
