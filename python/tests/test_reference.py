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
import scipy.sparse

import directional as dr

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"
ATOL = 1e-10


def assert_matches(pyArray, cppArray):
    np.testing.assert_allclose(np.asarray(pyArray, dtype=float).reshape(cppArray.shape), cppArray, atol=ATOL)


def assert_sparse_matches(pyMatrix, cpp, name):
    entries = cpp(name)
    cppMatrix = scipy.sparse.csc_matrix((entries[:, 2], (entries[:, 0].astype(int), entries[:, 1].astype(int))),
                                        shape=tuple(cpp(name + "_shape").astype(int).ravel()))
    assert pyMatrix.shape==cppMatrix.shape
    assert abs(pyMatrix-cppMatrix).max()<=ATOL


def test_mesh_matches_cpp(cpp_reference):
    cpp = cpp_reference("power_field")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    for name in ("EV", "EF", "FE", "faceNormals", "vertexNormals", "faceAreas", "midEdges", "FEs"):
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
    for name in ("sources", "normals", "adjSpaces", "cycleSources", "avgAdjLength", "cycleCurvatures"):
        assert_matches(getattr(tb, name), cpp(bundleName + "_" + name))
    for name in ("oneRing", "innerAdjacencies"):
        assert_matches(getattr(tb, name), cpp(bundleName + "_" + name))
    assert_matches(np.column_stack([tb.connection.real, tb.connection.imag]), cpp(bundleName + "_connection"))
    for name in ("connectionMass", "tangentSpaceMass"):
        assert_sparse_matches(getattr(tb, name), cpp, bundleName + "_" + name)


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


@pytest.mark.parametrize("name", ["fertility", "lilium"])
def test_topology_matches_cpp(cpp_reference, name):
    cpp = cpp_reference("topology")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / (name + ".obj")))
    assert_matches(mesh.innerEdges, cpp(name + "_innerEdges"))
    assert_matches(mesh.numGenerators, cpp(name + "_numGenerators"))
    assert_matches(len(mesh.boundaryLoops), cpp(name + "_numBoundaryLoops"))
    for i, loop in enumerate(mesh.boundaryLoops):
        assert_matches(loop, cpp(name + "_boundaryLoop" + str(i)))
    for array in ("TT", "EFi", "VE", "VF", "isBoundaryVertex", "GaussianCurvature", "FBx", "VBy"):
        assert_matches(getattr(mesh, array), cpp(name + "_" + array))
    assert_matches(np.column_stack([mesh.HV, mesh.HE, mesh.HF, mesh.nextH, mesh.twinH]), cpp(name + "_halfedges"))


@pytest.mark.parametrize("name", ["fertility", "lilium"])
def test_cycles_match_cpp(cpp_reference, name):
    cpp = cpp_reference("topology")
    ftb = dr.PCFaceTangentBundle(dr.readOBJ(str(TUTORIAL_DATA_PATH / (name + ".obj"))))
    assert_matches(ftb.local2Cycle, cpp(name + "_local2Cycle"))
    cycleEntries = cpp(name + "_cycles")
    cppCycles = scipy.sparse.csc_matrix((cycleEntries[:, 2], (cycleEntries[:, 0].astype(int), cycleEntries[:, 1].astype(int))), shape=ftb.cycles.shape)
    assert abs(ftb.cycles-cppCycles).max()==0.0


def test_prescribed_file_singularities_match_cpp(cpp_reference):
    cpp = cpp_reference("topology")
    ftb = dr.PCFaceTangentBundle(dr.readOBJ(str(TUTORIAL_DATA_PATH / "fertility.obj")))
    field = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, 2)
    dr.read_singularities(str(TUTORIAL_DATA_PATH / "fertility.sings"), field)
    cycleIndices = np.zeros(ftb.cycles.shape[0], dtype=int)
    cycleIndices[ftb.local2Cycle[field.singLocalCycles]] = field.singIndices
    _, linfError = dr.index_prescription(cycleIndices, 2, 0.0, field)
    assert_matches(field.extField, cpp("prescribed_extField"))
    assert_matches(linfError, cpp("prescribed_linfError"))


@pytest.mark.parametrize("name", ["spherers", "lilium"])
def test_fem_matrices_match_cpp(cpp_reference, name):
    cpp = cpp_reference("fem")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / (name + ".obj")))
    for isIntrinsic in (False, True):
        prefix = name + ("_intrinsic_" if isIntrinsic else "_extrinsic_")
        assert_sparse_matches(dr.conf_gradient_matrix_2D(mesh, isIntrinsic, 2), cpp, prefix + "conf_gradient")
        assert_sparse_matches(dr.non_conf_gradient_matrix_2D(mesh, isIntrinsic, 2), cpp, prefix + "non_conf_gradient")
        assert_sparse_matches(dr.curl_matrix_2D(mesh, isIntrinsic, 2), cpp, prefix + "curl")
        assert_sparse_matches(dr.div_matrix_2D(mesh, isIntrinsic, 2), cpp, prefix + "div")
        assert_sparse_matches(dr.face_vector_rotation_matrix_2D(mesh, isIntrinsic, 2), cpp, prefix + "rotation")
    assert_sparse_matches(dr.face_mass_matrix_2D(mesh, False, 2), cpp, name + "_face_mass")
    assert_sparse_matches(dr.face_mass_matrix_2D(mesh, True), cpp, name + "_inverse_face_mass")
    assert_sparse_matches(dr.edge_diamond_mass_matrix_2D(mesh), cpp, name + "_edge_diamond_mass")
    assert_sparse_matches(dr.face_extrinsic_to_intrinsic_matrix_2D(mesh, 2), cpp, name + "_extrinsic_to_intrinsic")
    assert_sparse_matches(dr.face_intrinsic_to_extrinsic_matrix_2D(mesh), cpp, name + "_intrinsic_to_extrinsic")


@pytest.mark.parametrize("name", ["spherers", "lilium"])
def test_dec_operators_match_cpp(cpp_reference, name):
    cpp = cpp_reference("dec")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / (name + ".obj")))
    assert_sparse_matches(dr.d0_matrix(mesh), cpp, name + "_d0")
    assert_sparse_matches(dr.d1_matrix(mesh), cpp, name + "_d1")
    assert_sparse_matches(dr.linear_whitney_mass_matrix(mesh), cpp, name + "_whitney_mass")
    for circumcenter in (True, False):
        h1, invh1 = dr.hodge_star_1_matrix(mesh, circumcenter)
        center = "_circumcenter" if circumcenter else "_barycenter"
        assert_sparse_matches(h1, cpp, name + center + "_hodge_star_1")
        assert_sparse_matches(invh1, cpp, name + center + "_inv_hodge_star_1")
    h2, invh2 = dr.hodge_star_2_matrix(mesh)
    assert_sparse_matches(h2, cpp, name + "_hodge_star_2")
    assert_sparse_matches(invh2, cpp, name + "_inv_hodge_star_2")


def test_cochain_functions_match_cpp(cpp_reference):
    cpp = cpp_reference("cochain")
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "torus.obj"))
    d0, d1 = dr.d0_matrix(mesh), dr.d1_matrix(mesh)
    h1, _ = dr.hodge_star_1_matrix(mesh)
    h2, _ = dr.hodge_star_2_matrix(mesh)
    oneForm = np.sin(mesh.midEdges[:, 0])+np.cos(mesh.midEdges[:, 1])
    for prefix, result in (("exact", dr.project_exact(d0, h1, oneForm)), ("gauge", dr.project_exact(d1, h1, d1@oneForm, gaugeFixing=True))):
        assert_matches(result[0], cpp(prefix + "_prevCochain"))
        assert_matches(result[1], cpp(prefix + "_exactCochain"))
    exactCochain, coexactCochain, harmCochain, _, _ = dr.hodge_decomposition(d0, d1, h1, h2, oneForm, 1)
    for name, value in (("exactCochain", exactCochain), ("coexactCochain", coexactCochain), ("harmCochain", harmCochain)):
        assert_matches(value, cpp("hodge_" + name))
    harmBasis = dr.cohomology_basis(d0, d1, h1, 2)
    np.testing.assert_allclose(harmBasis@(harmBasis.T@h1@oneForm), cpp("harmonic_projection").ravel(), atol=1e-6)


def test_integration_matches_cpp(cpp_reference):
    cpp = cpp_reference("integration")
    mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "vase.off"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "vase-4.rawfield"), dr.PCFaceTangentBundle(mesh))
    dr.principal_matching(rawField)
    intData = dr.IntegrationData(rawField.N)
    intData.integralSeamless = True
    for name, roundSeams in (("seams", True), ("sings", False)):
        intData.roundSeams = roundSeams
        meshCut, combedField = dr.setup_integration(rawField, intData)
        NFunction, NCornerFunctions = dr.integrate(combedField, intData, meshCut)
        for array, value in (("cutV", meshCut.V), ("cutF", meshCut.F), ("face2cut", intData.face2cut),
                             ("constrainedVertices", intData.constrainedVertices), ("matching", combedField.matching),
                             ("extField", combedField.extField), ("NFunction", NFunction), ("NCornerFunctions", NCornerFunctions),
                             ("nVertexFunction", intData.nVertexFunction)):
            assert_matches(value, cpp(name + "_" + array))
        isoV, isoE, _, _, funcNum = dr.branched_isolines(meshCut.V, meshCut.F, NFunction)
        for array, value in (("isoV", isoV), ("isoE", isoE), ("funcNum", funcNum)):
            assert_matches(value, cpp(name + "_" + array))


@pytest.mark.skipif(not hasattr(dr, "mesher"), reason="the module was built without GMP, so without meshing")
def test_meshing_matches_cpp(cpp_reference, gmp_flags):
    cpp = cpp_reference("meshing", gmp_flags)
    mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "vase.off"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "vase-4.rawfield"), dr.PCFaceTangentBundle(mesh))
    dr.principal_matching(rawField)
    intData = dr.IntegrationData(rawField.N)
    meshCut, combedField = dr.setup_integration(rawField, intData)
    intData.integralSeamless, intData.roundSeams = True, False
    dr.integrate(combedField, intData, meshCut)
    V, D, F = dr.mesher(mesh, dr.setup_mesher(meshCut, intData))
    assert cpp("success")[0, 0]==1
    assert_matches(V, cpp("VPolyMesh"))
    assert_matches(D, cpp("DPolyMesh"))
    #the C++ output leaves the entries past each face's valence uninitialized
    cppF = cpp("FPolyMesh")
    assert F.shape==cppF.shape
    for f in range(len(D)):
        np.testing.assert_array_equal(F[f, :D[f]], cppF[f, :D[f]])
