# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Compares each binding with the same computation in C++ (the programs in tests/reference), which catches wrong argument
# passing, defaults and array layouts. Both sides run the same library code, so library bugs are left to the other tests.
from pathlib import Path

import numpy as np
import pytest

import directional as dr

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"
ATOL = 1e-10


def assert_matches(pyArray, cppArray):
    np.testing.assert_allclose(np.asarray(pyArray, dtype=float).reshape(cppArray.shape), cppArray, atol=ATOL)


def test_mesh_matches_cpp(cpp_reference):
    cpp = cpp_reference("power_field")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    for name in ("EV", "EF", "FE", "faceNormals", "vertexNormals", "faceAreas"):
        assert_matches(getattr(mesh, name), cpp(name))


@pytest.mark.parametrize("bundleName, TangentBundle", [("face", dr.PCFaceTangentBundle), ("vertex", dr.IntrinsicVertexTangentBundle)])
@pytest.mark.parametrize("alignWeight", [-1.0, 1.0])
def test_power_field_matches_cpp(cpp_reference, tmp_path, bundleName, TangentBundle, alignWeight):
    cpp = cpp_reference("power_field")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    tb = TangentBundle(mesh)
    constSpaces = np.array([0, 100, 500])
    constVectors = mesh.V[mesh.F[constSpaces, 1]]-mesh.V[mesh.F[constSpaces, 0]]
    rawField = dr.power_to_raw(dr.power_field(tb, constSpaces, constVectors, np.full(3, alignWeight), 4), 4)
    dr.principal_matching(rawField)

    name = bundleName + ("_hard" if alignWeight<0.0 else "_soft")
    for array in ("extField", "matching", "effort", "singLocalCycles", "singIndices"):
        assert_matches(getattr(rawField, array), cpp(name + "_" + array))

    #the C++ file reads into the same field, and the Python file reads back into it
    cppField = dr.read_raw_field(str(cpp.outDir / (name + ".rawfield")), tb)
    dr.write_raw_field(str(tmp_path / "py.rawfield"), rawField, True)
    for otherField in (cppField, dr.read_raw_field(str(tmp_path / "py.rawfield"), tb)):
        assert_matches(otherField.extField, rawField.extField)


def test_read_singularities_matches_cpp(cpp_reference):
    cpp = cpp_reference("read_singularities")
    mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "bumpy.off"))
    field = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "bumpy.rawfield"), dr.PCFaceTangentBundle(mesh))
    dr.read_singularities(str(TUTORIAL_DATA_PATH / "bumpy.sings"), field)
    assert_matches(field.singLocalCycles, cpp("singLocalCycles"))
    assert_matches(field.singIndices, cpp("singIndices"))
