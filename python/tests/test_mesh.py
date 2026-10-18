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
