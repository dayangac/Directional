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


@pytest.mark.parametrize("TangentBundle", [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle])
def test_bundle_spaces(sphere, TangentBundle):
    tb = TangentBundle(sphere)
    numSpaces = len(sphere.F) if TangentBundle is dr.PCFaceTangentBundle else len(sphere.V)
    assert tb.numSpaces==numSpaces
    assert tb.sources.shape==tb.normals.shape==(numSpaces, 3)


def test_hard_constraints_are_met(sphere):
    constFaces = np.array([0, 100, 500])
    constVectors = sphere.V[sphere.F[constFaces, 1]]-sphere.V[sphere.F[constFaces, 0]]
    ftb = dr.PCFaceTangentBundle(sphere)
    powerField = dr.power_field(ftb, constFaces, constVectors, -np.ones(len(constFaces)), N)
    rawField = dr.power_to_raw(powerField, N)

    candidates = rawField.extField[constFaces].reshape(len(constFaces), N, 3)
    dist = np.linalg.norm(candidates-constVectors[:, None, :], axis=2).min(axis=1)
    np.testing.assert_allclose(dist, 0, atol=1e-8)


@pytest.mark.parametrize("TangentBundle", [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle])
def test_raw_field_is_rotationally_symmetric(sphere, TangentBundle):
    powerField = dr.power_field(TangentBundle(sphere), *NO_CONSTRAINTS, N)
    rawField = dr.power_to_raw(powerField, N)
    assert rawField.fieldType==dr.fieldTypeEnum.RAW_FIELD
    assert rawField.intField.shape==(powerField.intField.shape[0], 2*N)

    complexField = rawField.intField[:, 0::2]+1j*rawField.intField[:, 1::2]
    np.testing.assert_allclose(complexField[:, 1:], complexField[:, :-1]*np.exp(2j*np.pi/N), atol=1e-10)


@pytest.mark.parametrize("fileName", ["spherers.obj", "torus.obj"])
@pytest.mark.parametrize("TangentBundle", [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle])
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
