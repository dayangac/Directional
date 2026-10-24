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


#a disc with a boundary, and a closed torus whose handle has to be cut open as well
@pytest.fixture(scope="module", params=[("vase.off", "vase-4.rawfield"), ("torus.obj", "torus_twosings.rawfield")])
def whole(request):
    meshFile, rawFieldFile = request.param
    mesh = (dr.readOFF if meshFile.endswith(".off") else dr.readOBJ)(str(TUTORIAL_DATA_PATH / meshFile))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / rawFieldFile), dr.PCFaceTangentBundle(mesh))
    dr.principal_matching(rawField)
    return mesh, rawField


@pytest.fixture(scope="module")
def setup(whole):
    mesh, rawField = whole
    intData = dr.IntegrationData(rawField.N)
    meshCut, combedField = dr.setup_integration(rawField, intData)
    return intData, meshCut, combedField


def seams(mesh, intData):
    return np.unique(mesh.FE[intData.face2cut!=0])


def boundary_edges(mesh):
    return np.flatnonzero(np.any(mesh.EF<0, axis=1))


def integer_variables(intData, values):
    return np.concatenate([values[intData.n*i:intData.n*(i+1)] for i in intData.integerVars])


def test_integration_data_defaults():
    for N in (2, 3, 4, 7):
        intData = dr.IntegrationData(N)
        n = N//2 if N%2==0 else N
        assert (intData.N, intData.n)==(N, n)
        expected = np.vstack([np.eye(n), -np.eye(n)]) if N%2==0 else np.eye(N)
        np.testing.assert_array_equal(intData.linRed, expected)
        np.testing.assert_array_equal(intData.periodMat, np.eye(n))
    assert intData.lengthRatio==0.02
    assert (intData.integralSeamless, intData.roundSeams, intData.verbose)==(False, True, False)


def test_triangular_symmetry_sums_each_triple_to_zero():
    for N, n in ((3, 2), (6, 2), (9, 6), (12, 4)):
        intData = dr.IntegrationData(N)
        intData.set_triangular_symmetry(N)
        assert intData.linRed.shape==(N, n) and intData.n==n
        np.testing.assert_array_equal(intData.linRed.reshape(3, N//3, n).sum(axis=0), 0)
        np.testing.assert_array_equal(intData.periodMat, np.eye(n))
        if N%2==0:
            np.testing.assert_array_equal(intData.linRed[N//2:], -intData.linRed[:N//2])


def test_set_linear_reduction():
    intData = dr.IntegrationData(4)
    linRed, periodMat = np.array([[1], [1], [1]]), np.array([[2]])
    intData.set_linear_reduction(linRed, periodMat)
    assert (intData.N, intData.n)==(3, 1)
    np.testing.assert_array_equal(intData.linRed, linRed)
    np.testing.assert_array_equal(intData.periodMat, periodMat)


def test_default_period_matrix_is_the_identity():
    intData = dr.IntegrationData(4)
    intData.set_linear_reduction(np.array([[1], [1], [1]]), np.array([[2]]))
    intData.set_default_period_matrix(1)
    np.testing.assert_array_equal(intData.periodMat, np.eye(1))


def test_integration_data_rejects_bad_input():
    with pytest.raises(ValueError):
        dr.IntegrationData(0)
    with pytest.raises(ValueError):
        dr.IntegrationData(3).set_sign_symmetry(3)
    with pytest.raises(ValueError):
        dr.IntegrationData(4).set_sign_symmetry(6)
    with pytest.raises(ValueError):
        dr.IntegrationData(4).set_triangular_symmetry(4)
    with pytest.raises(ValueError):
        dr.IntegrationData(4).set_linear_reduction(np.eye(4, 2, dtype=int), np.eye(3, dtype=int))
    with pytest.raises(ValueError):
        dr.IntegrationData(4).set_default_period_matrix(3)


def test_cut_mesh_is_a_face_corresponding_disc(whole, setup):
    mesh, _ = whole
    _, meshCut, _ = setup
    assert len(meshCut.F)==len(mesh.F)
    np.testing.assert_array_equal(meshCut.V[meshCut.F], mesh.V[mesh.F])
    assert meshCut.eulerChar==1
    assert len(meshCut.boundaryLoops)==1


def test_seams_are_cut_open(whole, setup):
    mesh, _ = whole
    intData, meshCut, _ = setup
    assert np.all(np.isin(seams(mesh, intData), mesh.innerEdges))
    assert len(boundary_edges(meshCut))==2*len(seams(mesh, intData))+len(boundary_edges(mesh))


def test_combed_field_is_trivially_matched_off_seams(whole, setup):
    mesh, rawField = whole
    intData, _, combedField = setup
    offSeams = np.setdiff1d(mesh.innerEdges, seams(mesh, intData))
    np.testing.assert_array_equal(combedField.matching[offSeams], 0)
    np.testing.assert_array_equal(combedField.singLocalCycles, rawField.singLocalCycles)
    #every singularity is on the cut
    assert np.all(np.isin(rawField.singLocalCycles, mesh.EV[seams(mesh, intData)]))


#the transitions around a vertex cancel unless it is singular or a node of the cut graph
def test_constrained_vertices_are_singularities_and_cut_nodes(whole, setup):
    mesh, rawField = whole
    intData, _, _ = setup
    seamValence = np.bincount(mesh.EV[seams(mesh, intData)].ravel(), minlength=len(mesh.V))
    boundaryVertices = np.concatenate([[]]+mesh.boundaryLoops).astype(int)
    cutNodes = np.setdiff1d(np.flatnonzero(seamValence>2), boundaryVertices)
    np.testing.assert_array_equal(np.flatnonzero(intData.constrainedVertices), np.union1d(rawField.singLocalCycles, cutNodes))


@pytest.fixture(scope="module")
def integrated(setup):
    intData, meshCut, combedField = setup
    intData.integralSeamless = True
    NFunction, NCornerFunctions = dr.integrate(combedField, intData, meshCut)
    return NFunction, NCornerFunctions, intData.nVertexFunction.copy()


def test_integrate_is_sign_symmetric_per_corner(setup, integrated):
    _, meshCut, combedField = setup
    NFunction, NCornerFunctions, _ = integrated
    N = combedField.N
    assert NFunction.shape==(len(meshCut.V), N)
    np.testing.assert_array_equal(NFunction[:, N//2:], -NFunction[:, :N//2])
    np.testing.assert_array_equal(NCornerFunctions.reshape(-1, 3, N), NFunction[meshCut.F])


def test_function_gradients_follow_the_field(setup, integrated):
    _, meshCut, combedField = setup
    NFunction, _, _ = integrated
    G = dr.conf_gradient_matrix_2D(meshCut)
    for k in range(combedField.N):
        gradient = (G@NFunction[:, k]).reshape(-1, 3)
        vectors = combedField.extField[:, 3*k:3*k+3]
        cosines = np.sum(gradient*vectors, axis=1)/(np.linalg.norm(gradient, axis=1)*np.linalg.norm(vectors, axis=1))
        assert np.mean(cosines)>0.95


def test_integral_seamless_rounds_the_seam_translations(whole, setup, integrated):
    _, rawField = whole
    _, _, nVertexFunction = integrated
    translations = integer_variables(setup[0], nVertexFunction)
    assert len(translations)>0
    np.testing.assert_allclose(translations, np.round(translations), atol=1e-10)
    intData = dr.IntegrationData(rawField.N)
    meshCut, combedField = dr.setup_integration(rawField, intData)
    dr.integrate(combedField, intData, meshCut)
    translations = integer_variables(intData, intData.nVertexFunction)
    assert np.max(np.abs(translations-np.round(translations)))>1e-3


def test_rounding_singularities(whole):
    _, rawField = whole
    intData = dr.IntegrationData(rawField.N)
    intData.integralSeamless, intData.roundSeams = True, False
    meshCut, combedField = dr.setup_integration(rawField, intData)
    dr.integrate(combedField, intData, meshCut)
    assert len(intData.singularIndices)==intData.n*len(rawField.singLocalCycles)
    #all singularities are rounded, and then the remaining seams (around handles) as well
    for values in (intData.nVertexFunction[intData.singularIndices], integer_variables(intData, intData.nVertexFunction)):
        np.testing.assert_allclose(values, np.round(values), atol=1e-10)


def test_triangular_symmetry_of_integrated_functions(whole):
    mesh, _ = whole
    ftb = dr.PCFaceTangentBundle(mesh)
    N = 6
    rawField = dr.power_to_raw(dr.power_field(ftb, np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), N), N)
    dr.principal_matching(rawField)
    intData = dr.IntegrationData(N)
    intData.set_triangular_symmetry(N)
    meshCut, combedField = dr.setup_integration(rawField, intData)
    NFunction, _ = dr.integrate(combedField, intData, meshCut)
    np.testing.assert_allclose(NFunction[:, 0::2].sum(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(NFunction[:, 3:], -NFunction[:, :3], atol=1e-10)


def test_integration_rejects_bad_input(whole, setup):
    mesh, rawField = whole
    _, meshCut, combedField = setup
    with pytest.raises(ValueError):
        dr.setup_integration(rawField, dr.IntegrationData(3))
    with pytest.raises(ValueError):
        dr.setup_integration(dr.CartesianField(rawField.tb, dr.fieldTypeEnum.RAW_FIELD, rawField.N), dr.IntegrationData(rawField.N))
    vertexField = dr.CartesianField(dr.IntrinsicVertexTangentBundle(mesh), dr.fieldTypeEnum.RAW_FIELD, rawField.N)
    with pytest.raises(ValueError):
        dr.setup_integration(vertexField, dr.IntegrationData(rawField.N))
    #not set up
    with pytest.raises(ValueError):
        dr.integrate(combedField, dr.IntegrationData(rawField.N), meshCut)
    #a different linear reduction than the one that was set up
    intData = dr.IntegrationData(rawField.N)
    meshCut, combedField = dr.setup_integration(rawField, intData)
    intData.set_linear_reduction(np.eye(rawField.N, dtype=int), np.eye(rawField.N, dtype=int))
    with pytest.raises(ValueError):
        dr.integrate(combedField, intData, meshCut)


def test_combed_field_keeps_its_bundle_alive(whole):
    mesh, _ = whole
    rawField = dr.power_to_raw(dr.power_field(dr.PCFaceTangentBundle(mesh), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 2), 2)
    dr.principal_matching(rawField)
    intData = dr.IntegrationData(rawField.N)
    meshCut, combedField = dr.setup_integration(rawField, intData)
    del rawField  #the only reference to its bundle
    NFunction, _ = dr.integrate(combedField, intData, meshCut)
    assert np.all(np.isfinite(NFunction))


def test_branched_isolines_are_at_integer_values(whole):
    mesh, _ = whole
    f = 20.0*mesh.V@np.array([0.3, -0.5, 0.8])
    isoV, isoE, isoOrigE, isoN, funcNum = dr.branched_isolines(mesh.V, mesh.F, np.column_stack([f, -f]))
    assert len(isoE)>0 and isoE.shape[1]==2 and isoOrigE.shape==(len(isoE), 3)
    #a linear function is linear along edges, so its isolines interpolate it exactly
    np.testing.assert_allclose(20.0*isoV@np.array([0.3, -0.5, 0.8]), np.round(20.0*isoV@np.array([0.3, -0.5, 0.8])), atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(isoN, axis=1), 1.0)
    #only the first half of a sign-symmetric function is traced
    np.testing.assert_array_equal(funcNum, 0)
    _, isoE, _, _, funcNum = dr.branched_isolines(mesh.V, mesh.F, np.column_stack([f, 2.0*f, -3.0*f]))
    np.testing.assert_array_equal(np.unique(funcNum), [0, 1, 2])
    with pytest.raises(ValueError):
        dr.branched_isolines(mesh.V, mesh.F, np.zeros((3, 2)))


def test_integration_matrices_rebuild_the_functions(setup, integrated):
    intData, _, combedField = setup
    NFunction, _, nVertexFunction = integrated
    vertex2Cut = intData.vertexTrans2CutMat@intData.linRedMat@intData.singIntSpanMat@intData.intSpanMat
    np.testing.assert_allclose((vertex2Cut@nVertexFunction).reshape(-1, combedField.N), NFunction, atol=1e-12)
    #the seamless constraints hold at the solution
    constraints = intData.constraintMat@intData.linRedMat@intData.singIntSpanMat@intData.intSpanMat
    np.testing.assert_allclose(constraints@nVertexFunction, 0.0, atol=1e-10)


def test_fixed_values_translate_the_functions(whole):
    _, rawField = whole
    NFunctions = []
    for shift in (0.0, 0.25):
        intData = dr.IntegrationData(rawField.N)
        meshCut, combedField = dr.setup_integration(rawField, intData)
        intData.fixedValues = intData.fixedValues+shift
        NFunctions.append(dr.integrate(combedField, intData, meshCut)[0])
    np.testing.assert_allclose(NFunctions[1]-NFunctions[0], np.tile([0.25, 0.25, -0.25, -0.25], (len(NFunctions[0]), 1)), atol=1e-10)
    with pytest.raises(ValueError):
        intData.fixedValues = np.zeros(3)


def test_cut_mesh_with_singularities_is_the_integration_cut(whole, setup):
    mesh, rawField = whole
    intData, _, _ = setup
    face2cut = dr.cut_mesh_with_singularities(mesh, rawField.singLocalCycles)
    np.testing.assert_array_equal(face2cut, intData.face2cut)
    if not mesh.boundaryLoops:
        #a cut graph that opens a closed surface of genus g into a disc has E-V = 2g-1
        seams = np.unique(mesh.FE[face2cut!=0])
        assert len(seams)-len(np.unique(mesh.EV[seams]))==2*mesh.numGenerators-1
    with pytest.raises(ValueError):
        dr.cut_mesh_with_singularities(mesh, np.array([len(mesh.V)]))


def test_branched_gradient_of_linear_functions(setup):
    _, meshCut, _ = setup
    A = np.array([0.3, -1.2, 0.7])
    G = dr.branched_gradient(meshCut, 2)
    assert G.shape==(6*len(meshCut.F), 2*len(meshCut.V))
    gradients = (G@np.column_stack([meshCut.V@A, -meshCut.V@A]).ravel()).reshape(-1, 2, 3)
    tangentialA = A-np.outer(meshCut.faceNormals@A, np.ones(3))*meshCut.faceNormals
    np.testing.assert_allclose(gradients[:, 0], tangentialA, atol=1e-12)
    np.testing.assert_allclose(gradients[:, 1], -tangentialA, atol=1e-12)


def test_isolines_are_the_first_branch(whole):
    mesh, _ = whole
    f = 20.0*mesh.V@np.array([0.3, -0.5, 0.8])
    isoV, isoE, isoOrigE, isoN = dr.isolines(mesh.V, mesh.F, f)
    np.testing.assert_allclose(20.0*isoV@np.array([0.3, -0.5, 0.8]), np.round(20.0*isoV@np.array([0.3, -0.5, 0.8])), atol=1e-8)
    branchedV, branchedE, _, _, _ = dr.branched_isolines(mesh.V, mesh.F, np.column_stack([f, -f]))
    np.testing.assert_array_equal(isoV, branchedV)
    np.testing.assert_array_equal(isoE, branchedE)
    with pytest.raises(ValueError):
        dr.isolines(mesh.V, mesh.F, f[:-1])
