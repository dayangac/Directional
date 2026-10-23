# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg

import directional as dr

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"


@pytest.fixture(scope="module")
def sphere():
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))


@pytest.fixture(scope="module")
def lilium():
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), dr.PCFaceTangentBundle(mesh))
    dr.principal_matching(rawField)
    return rawField


def gradient_field(mesh, f):
    #per-face gradient of the piecewise-linear interpolation of f, whose tangential component is continuous across edges
    grad = np.zeros((len(mesh.F), 3))
    for i in range(3):
        e = mesh.V[mesh.F[:, (i+2)%3]]-mesh.V[mesh.F[:, (i+1)%3]]
        grad += f[mesh.F[:, i], None]*np.cross(mesh.faceNormals, e)
    return grad/(2.0*mesh.faceAreas[:, None])


def test_combing_is_a_cyclic_shift_per_face(lilium):
    combedField = dr.combing(lilium)
    N = lilium.N
    for f in range(0, lilium.tb.numSpaces, 97):
        shifts = [np.allclose(np.roll(lilium.intField[f], -2*s), combedField.intField[f]) for s in range(N)]
        assert any(shifts)


def test_combing_trivializes_matching_off_seams(lilium):
    combedField = dr.combing(lilium)
    assert np.count_nonzero(combedField.matching>0)<0.1*np.count_nonzero(lilium.matching>0)


def test_combing_keeps_singularities(lilium):
    combedField = dr.combing(lilium)
    np.testing.assert_array_equal(combedField.singLocalCycles, lilium.singLocalCycles)
    np.testing.assert_array_equal(combedField.singIndices, lilium.singIndices)


def test_combing_rejects_bad_input(sphere, lilium):
    unmatchedField = dr.CartesianField(lilium.tb, dr.fieldTypeEnum.RAW_FIELD, lilium.N)
    with pytest.raises(ValueError):
        dr.combing(unmatchedField)
    with pytest.raises(ValueError):
        dr.combing(lilium, np.zeros((lilium.tb.numSpaces, 2), dtype=int))
    vertexField = dr.power_to_raw(dr.power_field(dr.IntrinsicVertexTangentBundle(sphere), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 4), 4)
    dr.principal_matching(vertexField)
    with pytest.raises(ValueError):
        dr.combing(vertexField)


def test_curl_matching_of_gradient_field_is_curl_free(sphere):
    grad = gradient_field(sphere, sphere.V[:, 0]**2+np.sin(3.0*sphere.V[:, 1]))
    rawField = dr.CartesianField(dr.PCFaceTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, 2)
    rawField.set_extrinsic_field(np.hstack([grad, -grad]))
    curlNorm = dr.curl_matching(rawField)
    assert curlNorm.shape==(len(sphere.EV),)
    assert np.max(curlNorm)<1e-10*np.max(np.linalg.norm(grad, axis=1))
    assert np.sum(rawField.singIndices)==rawField.N*sphere.eulerChar


def test_curl_matching_rejects_bad_input(sphere, lilium):
    with pytest.raises(ValueError):
        dr.curl_matching(dr.power_field(lilium.tb, np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0), 4))
    vertexField = dr.power_to_raw(dr.power_field(dr.IntrinsicVertexTangentBundle(sphere), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 4), 4)
    with pytest.raises(ValueError):
        dr.curl_matching(vertexField)


@pytest.fixture
def prescribedField(sphere):
    return dr.CartesianField(dr.PCFaceTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, 2)


def prescribed_indices(field, singCycles, singIndices):
    cycleIndices = np.zeros(len(field.tb.cycleCurvatures), dtype=int)
    cycleIndices[singCycles] = singIndices
    return cycleIndices


def test_index_prescription_places_singularities(sphere, prescribedField):
    rotationAngles, linfError = dr.index_prescription(prescribed_indices(prescribedField, [35, 36], [2, 2]), 2, 0.0, prescribedField)
    assert linfError<1e-8
    assert rotationAngles.shape==(len(sphere.EV),)
    dr.principal_matching(prescribedField)
    np.testing.assert_array_equal(prescribedField.singLocalCycles, [35, 36])
    np.testing.assert_array_equal(prescribedField.singIndices, [2, 2])


def test_index_prescription_reports_inconsistent_indices(prescribedField):
    _, linfError = dr.index_prescription(prescribed_indices(prescribedField, [], []), 2, 0.0, prescribedField)
    assert linfError>0.1


#globalRotation is the angle of the power vector u^N, so every directional rotates by globalRotation/N.
def test_index_prescription_global_rotation(prescribedField):
    cycleIndices = prescribed_indices(prescribedField, [35, 36], [2, 2])
    dr.index_prescription(cycleIndices, 2, 0.0, prescribedField)
    unrotated = prescribedField.intField[:, 0]+1j*prescribedField.intField[:, 1]
    dr.index_prescription(cycleIndices, 2, np.pi/8.0, prescribedField)
    rotated = prescribedField.intField[:, 0]+1j*prescribedField.intField[:, 1]
    np.testing.assert_allclose((rotated/unrotated)**2, np.exp(1j*np.pi/8.0), atol=1e-8)


def test_index_prescription_rejects_bad_input(prescribedField):
    with pytest.raises(ValueError):
        dr.index_prescription(np.zeros(3, dtype=int), 2, 0.0, prescribedField)
    with pytest.raises(ValueError):
        dr.index_prescription(prescribed_indices(prescribedField, [], []), 0, 0.0, prescribedField)


def test_effort_to_indices_recovers_the_singularities(lilium):
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), lilium.tb)
    dr.principal_matching(rawField)
    singLocalCycles, singIndices = rawField.singLocalCycles.copy(), rawField.singIndices.copy()
    rawField.set_singularities(np.empty(0, dtype=int), np.empty(0, dtype=int))
    dr.effort_to_indices(rawField)
    np.testing.assert_array_equal(rawField.singLocalCycles, singLocalCycles)
    np.testing.assert_array_equal(rawField.singIndices, singIndices)


def test_effort_to_indices_on_arrays(lilium):
    tb = lilium.tb
    indices = dr.effort_to_indices(tb.cycles, lilium.effort[tb.innerAdjacencies], tb.cycleCurvatures, lilium.N)
    assert indices.shape==(tb.cycles.shape[0],)
    np.testing.assert_array_equal(np.flatnonzero(indices[tb.local2Cycle]), lilium.singLocalCycles)
    np.testing.assert_array_equal(indices[tb.local2Cycle[lilium.singLocalCycles]], lilium.singIndices)
    with pytest.raises(ValueError):  #not naturally integer
        dr.effort_to_indices(tb.cycles, lilium.effort[tb.innerAdjacencies]+0.1, tb.cycleCurvatures, lilium.N)
    with pytest.raises(ValueError):
        dr.effort_to_indices(tb.cycles, lilium.effort, tb.cycleCurvatures, lilium.N)
    with pytest.raises(ValueError):  #no matching yet
        dr.effort_to_indices(dr.CartesianField(tb, dr.fieldTypeEnum.RAW_FIELD, lilium.N))


def test_project_curl_gives_the_closest_curl_free_field(lilium):
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), lilium.tb)
    dr.curl_matching(rawField)
    mesh = lilium.tb.mesh
    C = dr.curl_matrix_2D(mesh, N=rawField.N, matching=rawField.matching)
    assert np.max(np.abs(C@rawField.extField.ravel()))>1e-3
    curlFreeField = dr.project_curl(rawField)
    np.testing.assert_allclose(C@curlFreeField.extField.ravel(), 0.0, atol=1e-12)
    #a projection: a curl-free field stays the same, and the change is in the range of the (intrinsic) curl transpose,
    #so orthogonal to all curl-free fields
    np.testing.assert_allclose(dr.project_curl(curlFreeField).extField, curlFreeField.extField, atol=1e-12)
    change = (rawField.intField-curlFreeField.intField).ravel()
    Cintrinsic = dr.curl_matrix_2D(mesh, True, rawField.N, rawField.matching)
    multipliers = scipy.sparse.linalg.lsqr(Cintrinsic.T, change, atol=1e-14, btol=1e-14, iter_lim=100000)[0]
    assert np.linalg.norm(Cintrinsic.T@multipliers-change)<1e-8*np.linalg.norm(change)


def test_project_curl_with_an_objective(lilium):
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), lilium.tb)
    dr.curl_matching(rawField)
    numDofs = rawField.intField.size
    #the identity objective is the default
    identity = scipy.sparse.identity(numDofs, format="csc")
    np.testing.assert_allclose(dr.project_curl(rawField, objMatrix=identity, objRhs=rawField.flatten(True)).extField,
                               dr.project_curl(rawField).extField, atol=1e-10)
    with pytest.raises(ValueError):
        dr.project_curl(rawField, objMatrix=identity)
    with pytest.raises(ValueError):
        dr.project_curl(rawField, objMatrix=scipy.sparse.identity(3, format="csc"), objRhs=np.zeros(3))
    with pytest.raises(ValueError):  #no matching
        dr.project_curl(dr.CartesianField(lilium.tb, dr.fieldTypeEnum.RAW_FIELD, lilium.N))
