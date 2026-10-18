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

