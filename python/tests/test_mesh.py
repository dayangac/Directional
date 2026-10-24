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


@pytest.mark.parametrize("fileName, genus", [("spherers.obj", 0), ("torus.obj", 1), ("fertility.obj", 4)])
def test_closed_mesh_topology(fileName, genus):
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / fileName))
    assert mesh.numGenerators==genus
    assert mesh.boundaryLoops==[]
    np.testing.assert_array_equal(mesh.innerEdges, np.arange(len(mesh.EV)))


def test_boundary_loops_are_closed_boundary_vertex_chains():
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    boundaryEdges = np.flatnonzero((mesh.EF<0).any(axis=1))
    assert len(mesh.boundaryLoops)==1
    loop = np.array(mesh.boundaryLoops[0])
    assert len(loop)==len(boundaryEdges)
    loopEdges = {frozenset(e) for e in zip(loop, np.roll(loop, -1))}
    assert loopEdges=={frozenset(e) for e in mesh.EV[boundaryEdges]}
    np.testing.assert_array_equal(np.sort(np.concatenate([mesh.innerEdges, boundaryEdges])), np.arange(len(mesh.EV)))


def test_mid_edges_and_face_edge_signs():
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    np.testing.assert_allclose(mesh.midEdges, (mesh.V[mesh.EV[:, 0]]+mesh.V[mesh.EV[:, 1]])/2.0)
    #FEs orients each edge along its face, so the signed edge vectors of a face close up
    edgeVectors = mesh.V[mesh.EV[:, 1]]-mesh.V[mesh.EV[:, 0]]
    np.testing.assert_allclose(np.einsum("fi,fix->fx", mesh.FEs, edgeVectors[mesh.FE]), 0.0, atol=1e-12)
    assert set(np.unique(mesh.FEs))=={-1.0, 1.0}


def test_writeOBJ_writes_texture_coordinates(torus, tmp_path):
    fileName = str(tmp_path / "torus.obj")
    UV = torus.V[:, [0, 2]]
    assert dr.writeOBJ(fileName, torus, UV, torus.F, "squares.mtl", "material_0")
    lines = Path(fileName).read_text().splitlines()
    assert lines[:2]==["mtllib squares.mtl", "usemtl material_0"]
    vt = np.array([line.split()[1:] for line in lines if line.startswith("vt ")], dtype=float)
    np.testing.assert_allclose(vt, UV, atol=1e-5)  #written in the default stream precision
    assert next(line for line in lines if line.startswith("f "))=="f " + " ".join(str(i+1) + "/" + str(i+1) for i in torus.F[0])
    readMesh = dr.readOBJ(fileName)
    np.testing.assert_allclose(readMesh.V, torus.V, atol=1e-5)
    np.testing.assert_array_equal(readMesh.F, torus.F)


def test_writeOBJ_rejects_bad_input(torus, tmp_path):
    with pytest.raises(ValueError):
        dr.writeOBJ(str(tmp_path / "torus.obj"), torus, torus.V[:, :2], torus.F[:-1])
    with pytest.raises(ValueError):
        dr.writeOBJ(str(tmp_path / "torus.obj"), torus, torus.V[:10, :2], torus.F)
    with pytest.raises(RuntimeError):
        dr.writeOBJ(str(tmp_path / "missing" / "torus.obj"), torus, torus.V[:, :2], torus.F)


@pytest.fixture(scope="module", params=["spherers.obj", "lilium.obj"])
def mesh(request):
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / request.param))


def test_boundary_flags(mesh):
    boundaryEdges = np.flatnonzero(np.any(mesh.EF<0, axis=1))
    np.testing.assert_array_equal(mesh.isBoundaryEdge, np.isin(np.arange(len(mesh.EV)), boundaryEdges))
    np.testing.assert_array_equal(mesh.boundEdges, boundaryEdges)
    boundaryVertices = np.concatenate([[]]+mesh.boundaryLoops).astype(int)
    np.testing.assert_array_equal(mesh.isBoundaryVertex, np.isin(np.arange(len(mesh.V)), boundaryVertices))


def test_adjacency_is_padded_with_minus_one(mesh):
    np.testing.assert_array_equal(mesh.vertexValence, np.bincount(mesh.EV.ravel(), minlength=len(mesh.V)))
    numFaces = np.bincount(mesh.F.ravel(), minlength=len(mesh.V))
    for v in range(0, len(mesh.V), 7):
        valence = mesh.vertexValence[v]
        assert set(mesh.VE[v, :valence])==set(np.flatnonzero(np.any(mesh.EV==v, axis=1)))
        assert set(mesh.VF[v, :numFaces[v]])==set(np.flatnonzero(np.any(mesh.F==v, axis=1)))
        np.testing.assert_array_equal(mesh.VE[v, valence:], -1)
        np.testing.assert_array_equal(mesh.VF[v, numFaces[v]:], -1)


def test_face_adjacency_is_across_face_edges(mesh):
    #TT[f, j] is the face on the other side of FE[f, j] (-1 on the boundary), which is at position EFi of the edge in its faces
    EF = mesh.EF[mesh.FE]
    np.testing.assert_array_equal(mesh.TT, np.where(EF[:, :, 0]==np.arange(len(mesh.F))[:, None], EF[:, :, 1], EF[:, :, 0]))
    for side in (0, 1):
        faces = mesh.EF[:, side]
        np.testing.assert_array_equal(mesh.EFi[:, side]==-1, faces==-1)
        np.testing.assert_array_equal(mesh.FE[faces[faces>=0], mesh.EFi[faces>=0, side]], np.flatnonzero(faces>=0))


def test_halfedges(mesh):
    halfedges = np.arange(len(mesh.HV))
    nextH, prevH, twinH = mesh.nextH, mesh.prevH, mesh.twinH
    assert len(halfedges)==3*len(mesh.F)
    np.testing.assert_array_equal(nextH[nextH[nextH]], halfedges)
    np.testing.assert_array_equal(prevH[nextH], halfedges)
    np.testing.assert_array_equal(mesh.HF[nextH], mesh.HF)
    inner = twinH>=0
    np.testing.assert_array_equal(twinH[twinH[inner]], halfedges[inner])
    np.testing.assert_array_equal(mesh.HE[twinH[inner]], mesh.HE[inner])
    #a halfedge goes from HV to the HV of the next one, along its edge
    np.testing.assert_array_equal(np.sort(mesh.EV[mesh.HE], axis=1), np.sort(np.column_stack([mesh.HV, mesh.HV[nextH]]), axis=1))
    np.testing.assert_array_equal(mesh.HV[mesh.VH], np.arange(len(mesh.V)))
    np.testing.assert_array_equal(mesh.HF[mesh.FH], np.arange(len(mesh.F)))
    np.testing.assert_array_equal(mesh.HE[mesh.EH[:, 0]], np.arange(len(mesh.EV)))
    np.testing.assert_array_equal(mesh.EH[:, 1], twinH[mesh.EH[:, 0]])


def test_local_bases_are_orthonormal_tangent_frames(mesh):
    for Bx, By, normals in ((mesh.FBx, mesh.FBy, mesh.faceNormals), (mesh.VBx, mesh.VBy, mesh.vertexNormals)):
        np.testing.assert_allclose(np.cross(Bx, By), normals, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(Bx, axis=1), 1.0, atol=1e-10)


def test_gaussian_curvature_satisfies_gauss_bonnet(mesh):
    #the angle defects, with the boundary turning angles at boundary vertices
    assert np.sum(mesh.GaussianCurvature)==pytest.approx(2.0*np.pi*mesh.eulerChar, abs=1e-8)


def test_bounding_box(mesh):
    np.testing.assert_array_equal(mesh.minBox.ravel(), mesh.V.min(axis=0))
    np.testing.assert_array_equal(mesh.maxBox.ravel(), mesh.V.max(axis=0))


def test_dual_cycles_are_the_face_bundle_cycles(mesh):
    basisCycles, cycleCurvature, vertex2cycle, innerEdges = dr.dual_cycles(mesh)
    ftb = dr.PCFaceTangentBundle(mesh)
    assert abs(basisCycles-ftb.cycles).max()==0.0
    np.testing.assert_array_equal(cycleCurvature, ftb.cycleCurvatures)
    np.testing.assert_array_equal(vertex2cycle, ftb.local2Cycle)
    np.testing.assert_array_equal(innerEdges, mesh.innerEdges)
    #inner vertices, then the boundary loops, then 2g generators
    assert basisCycles.shape==(np.count_nonzero(mesh.isBoundaryVertex==0)+len(mesh.boundaryLoops)+2*max(mesh.numGenerators, 0), len(innerEdges))


def test_barycentric_subdivision(mesh):
    VFine, FFine = dr.barycentric_subdivision(mesh.V, mesh.F)
    np.testing.assert_array_equal(VFine[:len(mesh.V)], mesh.V)
    np.testing.assert_allclose(VFine[len(mesh.V):], mesh.barycenters, atol=1e-15)
    assert FFine.shape==(3*len(mesh.F), 3)
    #each face splits into three with the same orientation and a third of the area
    fineMesh = dr.TriMesh(VFine, FFine)
    np.testing.assert_allclose(fineMesh.faceAreas, np.repeat(mesh.faceAreas/3.0, 3), rtol=1e-10)
    np.testing.assert_allclose(fineMesh.faceNormals, np.repeat(mesh.faceNormals, 3, axis=0), atol=1e-10)
    assert fineMesh.eulerChar==mesh.eulerChar
