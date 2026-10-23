# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

import gc
from pathlib import Path

import numpy as np
import pytest

import directional as dr

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"
N = 4
NO_CONSTRAINTS = (np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))
BUNDLES = [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle, dr.ExtrinsicVertexTangentBundle]


def load(fileName):
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / fileName))


@pytest.fixture(scope="module")
def sphere():
    return load("spherers.obj")


def test_mesh_topology(sphere):
    assert sphere.V.shape==(642, 3)
    assert sphere.F.shape==(1280, 3)
    assert sphere.eulerChar==2
    assert load("torus.obj").eulerChar==0


def test_mesh_from_arrays_matches_file(sphere):
    mesh = dr.TriMesh(sphere.V, sphere.F)
    np.testing.assert_array_equal(mesh.EV, sphere.EV)
    np.testing.assert_allclose(mesh.faceNormals, sphere.faceNormals)


def test_array_properties_are_read_only(sphere):
    rawField = dr.power_to_raw(dr.power_field(dr.PCFaceTangentBundle(sphere), *NO_CONSTRAINTS, N), N)
    for array in (sphere.V, sphere.F, dr.PCFaceTangentBundle(sphere).sources, rawField.extField):
        with pytest.raises(ValueError):
            array[0] = 0


@pytest.mark.parametrize("badFaces", [
    lambda F, numV: F[:, :2],
    lambda F, numV: F+numV,
    lambda F, numV: F-1,
])
def test_mesh_rejects_bad_faces(sphere, badFaces):
    with pytest.raises(ValueError):
        dr.TriMesh(sphere.V, badFaces(sphere.F, len(sphere.V)))


def test_readOBJ_missing_file():
    with pytest.raises(RuntimeError):
        dr.readOBJ(str(TUTORIAL_DATA_PATH / "missing.obj"))


@pytest.mark.parametrize("TangentBundle", BUNDLES)
def test_bundle_spaces(sphere, TangentBundle):
    tb = TangentBundle(sphere)
    numSpaces = len(sphere.F) if TangentBundle is dr.PCFaceTangentBundle else len(sphere.V)
    assert tb.numSpaces==numSpaces
    assert tb.sources.shape==tb.normals.shape==(numSpaces, 3)
    assert tb.avgAdjLength==pytest.approx(sphere.avgEdgeLength)


@pytest.mark.parametrize("TangentBundle", BUNDLES)
def test_bundle_adjacency_is_the_dual_or_primal_edges(TangentBundle):
    #lilium has a boundary, where one of the adjacent faces is -1
    mesh = load("lilium.obj")
    expected = mesh.EF if TangentBundle is dr.PCFaceTangentBundle else mesh.EV
    np.testing.assert_array_equal(TangentBundle(mesh).adjSpaces, expected)


def test_field_refers_to_its_bundle(sphere):
    ftb = dr.PCFaceTangentBundle(sphere)
    rawField = dr.power_to_raw(dr.power_field(ftb, *NO_CONSTRAINTS, N), N)
    assert rawField.tb is ftb
    assert isinstance(dr.CartesianField(dr.IntrinsicVertexTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, N).tb,
                      dr.IntrinsicVertexTangentBundle)


def test_bundle_cycles_are_dual_elements(sphere):
    np.testing.assert_array_equal(dr.PCFaceTangentBundle(sphere).cycleSources, sphere.V)
    np.testing.assert_array_equal(dr.IntrinsicVertexTangentBundle(sphere).cycleSources, sphere.barycenters)


def test_cycle_curvatures_sum_to_euler_characteristic(sphere):
    for TangentBundle in BUNDLES:
        assert np.sum(TangentBundle(sphere).cycleCurvatures)==pytest.approx(2.0*np.pi*sphere.eulerChar)


#a local cycle around every vertex (face bundle) or face (vertex bundle), then for face bundles 2*genus generator cycles
#(vertex bundles have no generator cycles yet, see the TODO in IntrinsicVertexTangentBundle::init())
@pytest.mark.parametrize("TangentBundle", BUNDLES)
def test_bundle_cycles(TangentBundle):
    mesh = load("fertility.obj")
    tb = TangentBundle(mesh)
    isFaceBundle = TangentBundle is dr.PCFaceTangentBundle
    numLocalCycles = len(mesh.V) if isFaceBundle else len(mesh.F)
    assert tb.cycles.shape==(numLocalCycles+(2*mesh.numGenerators if isFaceBundle else 0), len(mesh.innerEdges))
    assert len(tb.cycleCurvatures)==tb.cycles.shape[0]
    np.testing.assert_array_equal(tb.local2Cycle, np.arange(numLocalCycles))


def test_boundary_vertices_map_to_their_boundary_cycle():
    mesh = load("lilium.obj")
    ftb = dr.PCFaceTangentBundle(mesh)
    boundaryVertices = np.unique(mesh.boundaryLoops[0])
    innerVertices = np.setdiff1d(np.arange(len(mesh.V)), boundaryVertices)
    assert len(np.unique(ftb.local2Cycle[innerVertices]))==len(innerVertices)
    assert np.all(ftb.local2Cycle[boundaryVertices]==ftb.cycles.shape[0]-1)


def test_local_cycles_cross_every_inner_edge_twice(sphere):
    localCycles = dr.PCFaceTangentBundle(sphere).cycles[:len(sphere.V)].toarray()
    np.testing.assert_array_equal(np.sum(localCycles, axis=0), 0)
    np.testing.assert_array_equal(np.sum(np.abs(localCycles), axis=0), 2)


def test_cycles_give_the_singularity_indices(sphere):
    ftb = dr.PCFaceTangentBundle(sphere)
    rawField = dr.power_to_raw(dr.power_field(ftb, np.array([0, 500]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([-1.0, -1.0]), N), N)
    dr.principal_matching(rawField)
    indices = (ftb.cycles@rawField.effort[sphere.innerEdges]+N*ftb.cycleCurvatures)/(2.0*np.pi)
    np.testing.assert_allclose(indices, np.round(indices), atol=1e-8)
    expected = np.zeros(len(indices))
    expected[ftb.local2Cycle[rawField.singLocalCycles]] = rawField.singIndices
    np.testing.assert_array_equal(np.round(indices), expected)


def test_extrinsic_vertex_field_is_tangent_to_vertex_normals(sphere):
    vtb = dr.ExtrinsicVertexTangentBundle(sphere)
    np.testing.assert_array_equal(vtb.normals, sphere.vertexNormals)
    rawField = dr.power_to_raw(dr.power_field(vtb, np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), N), N)
    vectors = rawField.extField.reshape(-1, N, 3)
    np.testing.assert_allclose(np.sum(vectors*sphere.vertexNormals[:, None, :], axis=2), 0.0, atol=1e-10)


def test_hard_constraints_are_met(sphere):
    constFaces = np.array([0, 100, 500])
    constVectors = sphere.V[sphere.F[constFaces, 1]]-sphere.V[sphere.F[constFaces, 0]]
    ftb = dr.PCFaceTangentBundle(sphere)
    powerField = dr.power_field(ftb, constFaces, constVectors, -np.ones(len(constFaces)), N)
    rawField = dr.power_to_raw(powerField, N)

    candidates = rawField.extField[constFaces].reshape(len(constFaces), N, 3)
    dist = np.linalg.norm(candidates-constVectors[:, None, :], axis=2).min(axis=1)
    np.testing.assert_allclose(dist, 0, atol=1e-8)


@pytest.mark.parametrize("TangentBundle", BUNDLES)
def test_raw_field_is_rotationally_symmetric(sphere, TangentBundle):
    powerField = dr.power_field(TangentBundle(sphere), *NO_CONSTRAINTS, N)
    rawField = dr.power_to_raw(powerField, N)
    assert rawField.fieldType==dr.fieldTypeEnum.RAW_FIELD
    assert rawField.intField.shape==(powerField.intField.shape[0], 2*N)

    complexField = rawField.intField[:, 0::2]+1j*rawField.intField[:, 1::2]
    np.testing.assert_allclose(complexField[:, 1:], complexField[:, :-1]*np.exp(2j*np.pi/N), atol=1e-10)


@pytest.mark.parametrize("fileName", ["spherers.obj", "torus.obj"])
@pytest.mark.parametrize("TangentBundle", BUNDLES)
def test_singularities_satisfy_poincare_hopf(fileName, TangentBundle):
    mesh = load(fileName)
    rawField = dr.power_to_raw(dr.power_field(TangentBundle(mesh), *NO_CONSTRAINTS, N), N)
    dr.principal_matching(rawField)
    assert len(rawField.matching)==len(mesh.EV)
    assert rawField.singIndices.sum()==N*mesh.eulerChar


def test_normalized_field_has_unit_vectors(sphere):
    powerField = dr.power_field(dr.PCFaceTangentBundle(sphere), *NO_CONSTRAINTS, N, normalizeField=True)
    np.testing.assert_allclose(np.linalg.norm(powerField.intField, axis=1), 1)


def test_power_field_types(sphere):
    powerField = dr.power_field(dr.PCFaceTangentBundle(sphere), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 4)
    assert powerField.fieldType==dr.fieldTypeEnum.POWER_FIELD
    assert dr.power_to_raw(powerField, 4).fieldType==dr.fieldTypeEnum.RAW_FIELD


@pytest.mark.parametrize("args", [
    (np.array([0, 1]), np.ones((1, 3)), -np.ones(2), N),
    (np.array([0]), np.ones((1, 2)), -np.ones(1), N),
    (np.array([10**6]), np.ones((1, 3)), -np.ones(1), N),
    (*NO_CONSTRAINTS, 0),
])
def test_power_field_rejects_bad_input(sphere, args):
    with pytest.raises(ValueError):
        dr.power_field(dr.PCFaceTangentBundle(sphere), *args)


def test_set_field_projects_between_representations(sphere):
    ftb = dr.PCFaceTangentBundle(sphere)
    rawField = dr.power_to_raw(dr.power_field(ftb, *NO_CONSTRAINTS, N), N)

    fromIntrinsic = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, N)
    fromIntrinsic.set_intrinsic_field(rawField.intField)
    np.testing.assert_allclose(fromIntrinsic.extField, rawField.extField, atol=1e-12)

    fromExtrinsic = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, N)
    fromExtrinsic.set_extrinsic_field(rawField.extField)
    np.testing.assert_allclose(fromExtrinsic.intField, rawField.intField, atol=1e-12)


def test_set_field_rejects_wrong_shape(sphere):
    rawField = dr.CartesianField(dr.PCFaceTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, N)
    with pytest.raises(ValueError):
        rawField.set_intrinsic_field(np.zeros((len(sphere.F), 2)))
    with pytest.raises(ValueError):
        rawField.set_extrinsic_field(np.zeros((len(sphere.F)-1, 3*N)))


def test_principal_matching_rejects_power_field(sphere):
    powerField = dr.power_field(dr.PCFaceTangentBundle(sphere), *NO_CONSTRAINTS, N)
    with pytest.raises(ValueError):
        dr.principal_matching(powerField)


def test_field_outlives_python_references_to_mesh_and_bundle():
    def make_field():
        ftb = dr.PCFaceTangentBundle(load("spherers.obj"))
        return dr.power_to_raw(dr.power_field(ftb, *NO_CONSTRAINTS, N), N)

    rawField = make_field()
    gc.collect()
    dr.principal_matching(rawField)
    assert rawField.singIndices.sum()==2*N


def test_raw_field_file_roundtrip(sphere, tmp_path):
    ftb = dr.PCFaceTangentBundle(sphere)
    rawField = dr.power_to_raw(dr.power_field(ftb, *NO_CONSTRAINTS, N), N)
    fileName = str(tmp_path / "field.rawfield")
    dr.write_raw_field(fileName, rawField, high_precision=True)

    loadedField = dr.read_raw_field(fileName, ftb)
    assert loadedField.N==N
    np.testing.assert_allclose(loadedField.extField, rawField.extField, atol=1e-12)


def test_read_singularities_matches_principal_matching():
    mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "bumpy.off"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "bumpy.rawfield"), dr.PCFaceTangentBundle(mesh))
    dr.read_singularities(str(TUTORIAL_DATA_PATH / "bumpy.sings"), rawField)
    singElements, singIndices = np.loadtxt(TUTORIAL_DATA_PATH / "bumpy.sings", skiprows=1, dtype=int).T
    np.testing.assert_array_equal(rawField.singLocalCycles, singElements)
    np.testing.assert_array_equal(rawField.singIndices, singIndices)

    dr.principal_matching(rawField)
    order = np.argsort(rawField.singLocalCycles)
    np.testing.assert_array_equal(rawField.singLocalCycles[order], np.sort(singElements))
    np.testing.assert_array_equal(rawField.singIndices[order], singIndices[np.argsort(singElements)])


def test_read_singularities_rejects_other_degree(sphere, tmp_path):
    fileName = tmp_path / "field.sings"
    fileName.write_text(f"{N+1} 1\n0 1\n")
    rawField = dr.CartesianField(dr.PCFaceTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, N)
    with pytest.raises(ValueError):
        dr.read_singularities(str(fileName), rawField)
    with pytest.raises(RuntimeError):
        dr.read_singularities(str(tmp_path / "missing.sings"), rawField)


def test_read_raw_field_missing_file(sphere):
    with pytest.raises(RuntimeError):
        dr.read_raw_field(str(TUTORIAL_DATA_PATH / "missing.rawfield"), dr.PCFaceTangentBundle(sphere))


def test_field_representations(sphere):
    rawField = dr.power_to_raw(dr.power_field(dr.PCFaceTangentBundle(sphere), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 4), 4)
    np.testing.assert_array_equal(rawField.get_complex_intrinsic_field(), rawField.intField[:, 0::2]+1j*rawField.intField[:, 1::2])
    np.testing.assert_array_equal(rawField.flatten(), rawField.extField.ravel())
    np.testing.assert_array_equal(rawField.flatten(isIntrinsic=True), rawField.intField.ravel())
    pvField = dr.raw_to_polyvector(rawField)
    assert pvField.get_complex_intrinsic_field().shape==(len(sphere.F), 4)
    #a power field has a single complex number per space
    powerField = dr.power_field(dr.PCFaceTangentBundle(sphere), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 4)
    np.testing.assert_array_equal(powerField.get_complex_intrinsic_field(), powerField.intField[:, [0]]+1j*powerField.intField[:, [1]])
    with pytest.raises(ValueError):
        pvField.flatten()


def test_set_singularities(sphere):
    rawField = dr.CartesianField(dr.PCFaceTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, 4)
    rawField.set_singularities(np.array([3, 70]), np.array([1, -1]))
    np.testing.assert_array_equal(rawField.singLocalCycles, [3, 70])
    np.testing.assert_array_equal(rawField.singIndices, [1, -1])
    with pytest.raises(ValueError):
        rawField.set_singularities(np.array([3]), np.array([1, 1]))
    with pytest.raises(ValueError):
        rawField.set_singularities(np.array([len(sphere.V)]), np.array([1]))


def test_rotation_to_raw_without_rotations_is_parallel(sphere):
    ftb = dr.PCFaceTangentBundle(sphere)
    rawField = dr.rotation_to_raw(ftb, np.zeros(len(sphere.EV)), N, 0.3)
    assert rawField.fieldType==dr.fieldTypeEnum.RAW_FIELD and rawField.N==N
    vectors = rawField.get_complex_intrinsic_field()
    np.testing.assert_allclose(np.abs(vectors), 1.0, atol=1e-12)
    np.testing.assert_allclose(vectors[:, 1:], vectors[:, :-1]*np.exp(2j*np.pi/N), atol=1e-12)
    #globalRotation is the angle of the power vector N*angle on the first face (see index_prescription)
    assert np.angle(vectors[0, 0]**N)==pytest.approx(0.3)
    dr.principal_matching(rawField)
    assert np.sum(rawField.singIndices)==N*sphere.eulerChar
    with pytest.raises(ValueError):
        dr.rotation_to_raw(ftb, np.zeros(3), N, 0.0)


def test_rotation_to_raw_gives_the_prescribed_field(sphere):
    ftb = dr.PCFaceTangentBundle(sphere)
    cycleIndices = np.zeros(ftb.cycles.shape[0], dtype=int)
    cycleIndices[ftb.local2Cycle[[0, 100]]] = N
    prescribedField = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, N)
    rotationAngles, _ = dr.index_prescription(cycleIndices, N, 0.0, prescribedField)
    np.testing.assert_allclose(dr.rotation_to_raw(ftb, rotationAngles, N, 0.0).extField, prescribedField.extField, atol=1e-12)


def test_singularities_file_roundtrip(sphere, tmp_path):
    fileName = str(tmp_path / "sphere.sings")
    dr.write_singularities(fileName, N, np.array([3, 70]), np.array([1, -1]))
    rawField = dr.CartesianField(dr.PCFaceTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, N)
    dr.read_singularities(fileName, rawField)
    np.testing.assert_array_equal(rawField.singLocalCycles, [3, 70])
    np.testing.assert_array_equal(rawField.singIndices, [1, -1])
    with pytest.raises(RuntimeError):
        dr.write_singularities(str(tmp_path / "missing" / "sphere.sings"), N, np.array([3]), np.array([1]))
    with pytest.raises(ValueError):
        dr.write_singularities(fileName, N, np.array([3]), np.array([1, 1]))
