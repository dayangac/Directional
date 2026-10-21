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


@pytest.fixture(scope="module", params=["spherers.obj", "lilium.obj"])
def mesh(request):
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / request.param))


@pytest.fixture(scope="module")
def flatMesh():
    return dr.readOFF(str(TUTORIAL_DATA_PATH / "flatmesh.off"))


def inner_vertices(mesh):
    return np.setdiff1d(np.arange(len(mesh.V)), np.concatenate([[]]+mesh.boundaryLoops).astype(int))


def test_differentials_form_an_exact_sequence(mesh):
    d0, d1 = dr.d0_matrix(mesh), dr.d1_matrix(mesh)
    assert d0.shape==(len(mesh.EV), len(mesh.V))
    assert d1.shape==(len(mesh.F), len(mesh.EV))
    values = np.random.default_rng(0).normal(size=len(mesh.V))
    np.testing.assert_allclose(d0@values, values[mesh.EV[:, 1]]-values[mesh.EV[:, 0]], atol=1e-14)
    np.testing.assert_array_equal(np.asarray(d1.tocsr()[np.repeat(np.arange(len(mesh.F)), 3), mesh.FE.ravel()]).reshape(-1, 3), mesh.FEs)
    assert abs(d1@d0).max()==0.0


def test_hodge_stars(mesh):
    for circumcenter in (True, False):
        h1, invh1 = dr.hodge_star_1_matrix(mesh, circumcenter)
        assert h1.shape==invh1.shape==(len(mesh.EV), len(mesh.EV))
        weights = h1.diagonal()
        np.testing.assert_allclose((h1@invh1).diagonal(), np.where(np.abs(weights)>10e-7, 1.0, 0.0), atol=1e-12)
    #the barycentric dual always has positive weights
    assert np.all(dr.hodge_star_1_matrix(mesh, False)[0].diagonal()>0.0)
    h2, invh2 = dr.hodge_star_2_matrix(mesh)
    np.testing.assert_allclose(h2.diagonal(), mesh.faceAreas)
    np.testing.assert_allclose(invh2.diagonal(), 1.0/mesh.faceAreas)


def test_whitney_and_cotangent_laplacians_agree(mesh):
    d0 = dr.d0_matrix(mesh)
    M1 = dr.linear_whitney_mass_matrix(mesh)
    assert abs(M1-M1.T).max()<1e-14
    h1, _ = dr.hodge_star_1_matrix(mesh)
    assert abs(d0.T@M1@d0-d0.T@h1@d0).max()<1e-10


def test_cotangent_laplacian_has_linear_precision(flatMesh):
    d0 = dr.d0_matrix(flatMesh)
    h1, _ = dr.hodge_star_1_matrix(flatMesh)
    linear = flatMesh.V@np.array([0.4, -1.1, 0.0])
    np.testing.assert_allclose((d0.T@h1@d0@linear)[inner_vertices(flatMesh)], 0.0, atol=1e-10)
