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


@pytest.fixture(scope="module")
def torus():
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / "torus.obj"))


def test_project_exact_reproduces_an_exact_cochain(mesh):
    d0 = dr.d0_matrix(mesh)
    h1, _ = dr.hodge_star_1_matrix(mesh)
    potential = np.sin(3.0*mesh.V[:, 0])
    prevCochain, exactCochain = dr.project_exact(d0, h1, d0@potential)
    np.testing.assert_allclose(exactCochain, d0@potential, atol=1e-8)
    np.testing.assert_allclose(prevCochain-potential, np.mean(prevCochain-potential), atol=1e-8)  #up to a constant


def test_project_exact_with_gauge_fixing_finds_the_minimal_potential(torus):
    d0, d1 = dr.d0_matrix(torus), dr.d1_matrix(torus)
    h1, _ = dr.hodge_star_1_matrix(torus)
    curl = torus.faceAreas*np.sin(2.0*torus.barycenters[:, 0])
    curl -= np.mean(curl)  #in the image of d1 on a closed mesh
    prevCochain, exactCochain = dr.project_exact(d1, h1, curl, gaugeFixing=True)
    np.testing.assert_allclose(exactCochain, curl, atol=1e-10)
    #the minimal-norm potential is orthogonal to the kernel of d1, and in particular to all exact 1-forms
    np.testing.assert_allclose(d0.T@h1@prevCochain, 0.0, atol=1e-10)


def harmonic_basis(torus):
    d0, d1 = dr.d0_matrix(torus), dr.d1_matrix(torus)
    h1, _ = dr.hodge_star_1_matrix(torus)
    bettiNumber = len(torus.EV)-(len(torus.V)-1)-(len(torus.F)-1)
    return dr.cohomology_basis(d0, d1, h1, bettiNumber)


def test_cohomology_basis_is_harmonic_and_orthonormal(torus):
    d0, d1 = dr.d0_matrix(torus), dr.d1_matrix(torus)
    h1, _ = dr.hodge_star_1_matrix(torus)
    harmBasis = harmonic_basis(torus)
    assert harmBasis.shape==(len(torus.EV), 2)
    scale = np.max(np.abs(harmBasis))
    np.testing.assert_allclose(d1@harmBasis, 0.0, atol=1e-6*scale)
    np.testing.assert_allclose(d0.T@h1@harmBasis, 0.0, atol=1e-6*scale)
    np.testing.assert_allclose(harmBasis.T@h1@harmBasis, np.eye(2), atol=1e-6)


def test_hodge_decomposition_recovers_its_parts(torus):
    d0, d1 = dr.d0_matrix(torus), dr.d1_matrix(torus)
    h1, _ = dr.hodge_star_1_matrix(torus)
    h2, _ = dr.hodge_star_2_matrix(torus)
    curl = torus.faceAreas*np.sin(2.0*torus.barycenters[:, 0])
    exactPart = d0@np.cos(3.0*torus.V[:, 1])
    coexactPart = dr.project_exact(d1, h1, curl-np.mean(curl), gaugeFixing=True)[0]
    harmPart = harmonic_basis(torus)[:, 0]
    exactCochain, coexactCochain, harmCochain, prevCochain, nextCochain = dr.hodge_decomposition(d0, d1, h1, h2, exactPart+coexactPart+harmPart, 1)
    for part, expected in ((exactCochain, exactPart), (coexactCochain, coexactPart), (harmCochain, harmPart)):
        np.testing.assert_allclose(part, expected, atol=1e-6*np.max(np.abs(expected)))
    np.testing.assert_allclose(exactCochain, d0@prevCochain, atol=1e-10)
    #coexactCochain = (-1)^k*M^{-1}*dNext^T*MNext*nextCochain, without inverting M (invHodgeStar zeroes its tiny weights)
    np.testing.assert_allclose(h1@coexactCochain, -d1.T@h2@nextCochain, atol=1e-10)
    for part1, part2 in ((exactCochain, coexactCochain), (exactCochain, harmCochain), (coexactCochain, harmCochain)):
        assert abs(part1@h1@part2)<1e-10


def test_cochain_functions_reject_mismatched_sizes(torus):
    d0, d1 = dr.d0_matrix(torus), dr.d1_matrix(torus)
    h1, _ = dr.hodge_star_1_matrix(torus)
    h2, _ = dr.hodge_star_2_matrix(torus)
    with pytest.raises(ValueError):
        dr.project_exact(d0, h1, np.zeros(len(torus.V)))
    with pytest.raises(ValueError):
        dr.project_exact(d1, h2, np.zeros(len(torus.F)), gaugeFixing=True)
    with pytest.raises(ValueError):
        dr.hodge_decomposition(d0, d1, h2, h2, np.zeros(len(torus.EV)), 1)
    with pytest.raises(ValueError):
        dr.cohomology_basis(d0, d1, h1, 0)
