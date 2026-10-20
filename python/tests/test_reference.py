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


def test_principal_curvatures_match_cpp(cpp_reference):
    cpp = cpp_reference("principal_curvatures")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "torus.obj"))
    for currMesh in (mesh, dr.TriMesh(mesh.V, mesh.F)):
        for name in ("minVertexPrincipalDirections", "maxVertexPrincipalDirections", "vertexPrincipalCurvatures"):
            assert_matches(getattr(currMesh, name), cpp(name))


#automatic seeds are left out, since they are random on every run
def test_streamlines_match_cpp(cpp_reference):
    cpp = cpp_reference("streamlines")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    ftb = dr.PCFaceTangentBundle(mesh)
    constVectors = (mesh.V[mesh.F[0, 1]]-mesh.V[mesh.F[0, 0]]).reshape(1, 3)
    rawField = dr.power_to_raw(dr.power_field(ftb, np.array([0]), constVectors, np.array([-1.0]), 3), 3, normalize=True)
    data, state = dr.StreamlineData(), dr.StreamlineState()
    dr.streamlines_init(rawField, np.array([0, 10, 500]), 3.0, data, state)
    for i in range(3):
        dr.streamlines_next(data, state, 0.5*mesh.avgEdgeLength)
    for array, value in (("samplePoints", data.samplePoints), ("segStart", state.segStart), ("segEnd", state.segEnd)):
        assert_matches(value, cpp(array))


@pytest.mark.parametrize("bundleName, TangentBundle", [("face", dr.PCFaceTangentBundle), ("vertex", dr.IntrinsicVertexTangentBundle),
                                                    ("extrinsic", dr.ExtrinsicVertexTangentBundle)])
def test_bundle_matches_cpp(cpp_reference, bundleName, TangentBundle):
    cpp = cpp_reference("bundle")
    tb = TangentBundle(dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj")))
    for name in ("sources", "normals", "cycleSources", "avgAdjLength", "cycleCurvatures"):
        assert_matches(getattr(tb, name), cpp(bundleName + "_" + name))


def test_extrinsic_power_field_matches_cpp(cpp_reference):
    vtb = dr.ExtrinsicVertexTangentBundle(dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj")))
    rawField = dr.power_to_raw(dr.power_field(vtb, np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 4), 4)
    assert_matches(rawField.extField, cpp_reference("bundle")("extrinsic_extField"))


def test_readDMAT_matches_cpp(cpp_reference):
    assert_matches(dr.readDMAT(str(TUTORIAL_DATA_PATH / "spheres_constFaces.dmat")), cpp_reference("read_dmat")("dmat"))


def test_combing_and_curl_matching_match_cpp(cpp_reference):
    cpp = cpp_reference("matching")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), dr.PCFaceTangentBundle(mesh))
    dr.principal_matching(rawField)
    combedField = dr.combing(rawField)
    curlNorm = dr.curl_matching(rawField)
    for name, field in (("combed", combedField), ("curl", rawField)):
        for array in ("extField", "matching", "singLocalCycles", "singIndices"):
            assert_matches(getattr(field, array), cpp(name + "_" + array))
    assert_matches(rawField.effort, cpp("curl_effort"))
    assert_matches(curlNorm, cpp("curl_curlNorm"))


def test_index_prescription_matches_cpp(cpp_reference):
    cpp = cpp_reference("index_prescription")
    ftb = dr.PCFaceTangentBundle(dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj")))
    field = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, 2)
    cycleIndices = np.zeros(len(ftb.cycleCurvatures), dtype=int)
    cycleIndices[[35, 36]] = 2
    rotationAngles, linfError = dr.index_prescription(cycleIndices, 2, np.pi/16.0, field)
    assert_matches(field.extField, cpp("extField"))
    assert_matches(rotationAngles, cpp("rotationAngles"))
    assert_matches(linfError, cpp("linfError"))


def pv_data(ftb, mesh):
    pvData = dr.PolyVectorData(ftb, 4)
    pvData.constSpaces = np.array([0, 0, 300, 900])
    edges = np.array([[0, 1], [0, 2], [0, 1], [1, 2]])
    F = mesh.F[pvData.constSpaces]
    constVectors = mesh.V[F[np.arange(4), edges[:, 1]]]-mesh.V[F[np.arange(4), edges[:, 0]]]
    pvData.constVectors = constVectors/np.linalg.norm(constVectors, axis=1)[:, None]
    pvData.wAlignment = np.ones(4)
    return pvData


@pytest.mark.parametrize("name", ["hard", "soft", "hard_rosy", "curl_projection"])
def test_polyvector_field_matches_cpp(cpp_reference, name):
    cpp = cpp_reference("polyvector")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    pvData = pv_data(dr.PCFaceTangentBundle(mesh), mesh)
    if name=="hard":
        pvData.wAlignment = np.full(4, -1.0)
    if name=="hard_rosy":
        pvData.wRoSy = -1.0
        pvData.initImplicitFactor = 0.2
        pvData.implicitScheduler = 1.0
    if name=="curl_projection":
        pvData.wRoSy = 5.0
    pvData.iterationMode = name in ("hard_rosy", "curl_projection")
    pvField = dr.polyvector_field(pvData)
    if name=="hard_rosy":
        dr.polyvector_iterate(pvData, pvField, [dr.hard_rosy], 5)
    if name=="curl_projection":
        dr.polyvector_iterate(pvData, pvField, [dr.soft_rosy, dr.curl_projection], 3)
    assert_matches(pvField.intField, cpp(name + "_intField"))
    assert_matches(dr.polyvector_to_raw(pvField).extField, cpp(name + "_extField"))


def test_raw_to_polyvector_matches_cpp(cpp_reference):
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), dr.PCFaceTangentBundle(mesh))
    assert_matches(dr.raw_to_polyvector(rawField).intField, cpp_reference("polyvector")("raw_to_polyvector_intField"))
