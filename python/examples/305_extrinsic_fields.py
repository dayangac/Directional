# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/305_ExtrinsicFields: a vertex-based field with an extrinsic connection, interpolated inside faces,
# with the exact locations of its singularities.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 1
interpResolution = 20
FACE_FIELD, VERTEX_FIELD, INTERPOLATED_FIELD = range(3)
viewingMode = FACE_FIELD


def normalized(v):
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a, b):
    return np.sum(a*b, axis=-1)


def apply(R, v):
    return np.einsum("...ij,...j->...i", R, v)


#The rotation matrices that rotate n1 to n2 in their common plane (batched directional::shortest_rotation_matrix(),
#without its case of opposite normals, which never happens between normals inside a triangle)
def shortest_rotation_matrix(n1, n2):
    u, v = normalized(n1), normalized(n2)
    w = np.cross(u, v)
    K = np.zeros(w.shape+(3,))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K = K-np.swapaxes(K, -1, -2)
    return np.eye(3)+K+K@K/(1.0+dot(u, v))[..., None, None]


#The normal interpolated at barycentric coordinates baryCoords (...x3) of a triangle with vertex normals n (3x3)
def compute_normal(baryCoords, n):
    return normalized(baryCoords@n)


#Interpolates the vertex vectors v (3x3) of a triangle with vertex normals n (3x3) at baryCoords (...x3), by averaging
#their N-th powers after transporting them to the interpolated normal
def interpolate_vector(baryCoords, n, v, N):
    np_ = compute_normal(baryCoords, n)
    R = shortest_rotation_matrix(n, np_[..., None, :])  #...x3x3x3: a rotation from each vertex normal
    Vxp = normalized(apply(R[..., 0, :, :], v[0]))
    Vyp = np.cross(np_, Vxp)
    Vx = normalized(v)
    Vy = np.cross(n, Vx)
    transVx = apply(R, Vx)
    conn = dot(transVx, Vxp[..., None, :])+1j*dot(transVx, Vyp[..., None, :])
    cv = dot(v, Vx)+1j*dot(v, Vy)
    cvp = np.sum(baryCoords*cv**N*conn**N, axis=-1)**(1.0/N)
    return cvp.real[..., None]*Vxp+cvp.imag[..., None]*Vyp


def barycentric_samples(interpResolution):
    n = interpResolution+1
    return np.array([(i/n, j/n, (n-i-j)/n) for i in range(n+1) for j in range(n+1-i)])


def rotate_around_axis(v, n, t):
    axis = normalized(n)
    return v*np.cos(t)+np.cross(axis, v)*np.sin(t)+axis*dot(axis, v)[..., None]*(1.0-np.cos(t))


#The holonomy of the extrinsic connection along the triangle i->ij->ki->i, and the index of the field around it
def loop_index(ni, nij, nki, vi, vij, vki):
    Riij, Rijki, Rkii = shortest_rotation_matrix(ni, nij), shortest_rotation_matrix(nij, nki), shortest_rotation_matrix(nki, ni)
    loopR = Rkii@Rijki@Riij
    w = np.array([loopR[2, 1]-loopR[1, 2], loopR[0, 2]-loopR[2, 0], loopR[1, 0]-loopR[0, 1]])
    holonomy = np.arctan2(0.5*dot(w, ni), 0.5*(np.trace(loopR)-1.0))

    def effort(R, n, vFrom, vTo):
        transV = R@vFrom
        return (N*np.arctan2(dot(n, np.cross(transV, vTo)), dot(transV, vTo))+np.pi)%(2.0*np.pi)-np.pi

    return (effort(Riij, nij, vi, vij)+effort(Rijki, nki, vij, vki)+effort(Rkii, ni, vki, vi)+N*holonomy)/(2.0*np.pi)


#Binary search for the barycentric coordinate b of corner i such that the loop through (b,1-b,0) and (b,0,1-b) no longer contains the singularity
def search_barycentric_line(n, v, i):
    j, k = (i+1)%3, (i+2)%3
    baryBegin, baryEnd = 0.0, 1.0
    while baryEnd-baryBegin>1e-7:
        currBaryLine = (baryBegin+baryEnd)/2.0
        coordSource, coordTarget = np.zeros(3), np.zeros(3)
        coordSource[[i, j, k]] = currBaryLine, 1.0-currBaryLine, 0.0
        coordTarget[[i, j, k]] = currBaryLine, 0.0, 1.0-currBaryLine
        index = loop_index(n[i], compute_normal(coordSource, n), compute_normal(coordTarget, n),
                           v[i], interpolate_vector(coordSource, n, v, N), interpolate_vector(coordTarget, n, v, N))
        if abs(index)<10e-5:
            baryEnd = currBaryLine
        else:
            baryBegin = currBaryLine
    return currBaryLine


def locate_singularities(vertexField):
    singLocations = np.zeros((len(vertexField.singIndices), 3))
    for s, singFace in enumerate(vertexField.singLocalCycles):
        n = mesh.vertexNormals[mesh.F[singFace]]
        v = normalized(vertexField.extField[mesh.F[singFace], :3])
        Bi, Bj = search_barycentric_line(n, v, 0), search_barycentric_line(n, v, 1)
        singLocations[s] = np.array([Bi, Bj, 1.0-Bi-Bj])@mesh.V[mesh.F[singFace]]
    return singLocations


def extrinsic_interpolation(vertexField, interpResolution, normalize):
    barySamples = barycentric_samples(interpResolution)
    newSources = np.zeros((len(mesh.F), len(barySamples), 3))
    newRawField = np.zeros((len(mesh.F), len(barySamples), 3*N))
    for i, face in enumerate(mesh.F):
        n = mesh.vertexNormals[face]
        np_ = compute_normal(barySamples, n)
        vp = interpolate_vector(barySamples, n, vertexField.extField[face, :3], N)
        if normalize:
            vp = normalized(vp)
        for k in range(N):
            newRawField[i, :, 3*k:3*k+3] = rotate_around_axis(vp, np_, k*2.0*np.pi/N)
        newSources[i] = barySamples@mesh.V[face]
    return newSources.reshape(-1, 3), newRawField.reshape(-1, 3*N)


def update_visualization():
    ps.get_point_cloud("Face-Based Field singularities").set_enabled(viewingMode==FACE_FIELD)
    ps.get_point_cloud("Exact Singularities").set_enabled(viewingMode==VERTEX_FIELD)
    for name, mode in (("Face-Based Field", FACE_FIELD), ("Vertex-Based Field", VERTEX_FIELD), ("Interpolated Field", INTERPOLATED_FIELD)):
        ps.get_point_cloud(name).set_enabled(viewingMode==mode)


def callbackFunc():
    global viewingMode
    psim.PushItemWidth(100)
    changed, viewingMode = psim.Combo("Viewing mode", viewingMode, ["Face-based field", "Extrinsic Vertex-based field", "Interpolated Field"])
    if changed:
        update_visualization()
    psim.PopItemWidth()


mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "bunny1k.obj"))
ftb = dr.PCFaceTangentBundle(mesh)
vtb = dr.ExtrinsicVertexTangentBundle(mesh)

constFaces = np.array([0])
constVertices = np.array([mesh.F[0, 1]])
constVectors = (mesh.V[mesh.F[0, 2]]-mesh.V[mesh.F[0, 1]]).reshape(1, 3)

normalizeField = True
powerVertexField = dr.power_field(vtb, constVertices, constVectors, np.full(constVertices.size, -1.0), N, normalizeField)
powerFaceField = dr.power_field(ftb, constFaces, constVectors, np.full(constFaces.size, -1.0), N, normalizeField)
rawFaceField = dr.power_to_raw(powerFaceField, N, normalizeField)
rawVertexField = dr.power_to_raw(powerVertexField, N, normalizeField)
dr.principal_matching(rawFaceField)
dr.principal_matching(rawVertexField)

normalizeInterpolation = True
newSources, newRawField = extrinsic_interpolation(rawVertexField, interpResolution, normalizeInterpolation)
singLocations = locate_singularities(rawVertexField)

ps.init()
viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(rawFaceField, "Face-Based Field")
viewer.set_cartesian_field(rawVertexField, "Vertex-Based Field")
viewer.set_raw_field(newSources, newRawField, "Interpolated Field", 0.3*mesh.avgEdgeLength/interpResolution)
#the exact singularities replace the ones at the vertices, as update_singularities_locations does in C++
ps.remove_point_cloud("Vertex-Based Field singularities")
viewer.update_singularities_locations(rawVertexField, singLocations, "Exact Singularities")
update_visualization()
ps.set_user_callback(callbackFunc)
ps.show()
