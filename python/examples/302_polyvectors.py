# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/302_PolyVectors: PolyVector fields aligned to the sharp edges of a mesh, with hard and soft alignment.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 4
smoothWeight, roSyWeight, alignWeight = 1.0, 1.0, 1.0
CONSTRAINTS, HARD_PRESCRIPTION, SOFT_PRESCRIPTION = range(3)
viewingMode = CONSTRAINTS

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "fandisk.off"))
ftb = dr.PCFaceTangentBundle(mesh)

#constraining both faces of every sharp edge to the edge direction
sharpEdges = np.flatnonzero(np.sum(mesh.faceNormals[mesh.EF[:, 0]]*mesh.faceNormals[mesh.EF[:, 1]], axis=1)<0.5)
edgeDirections = mesh.V[mesh.EV[sharpEdges, 0]]-mesh.V[mesh.EV[sharpEdges, 1]]
constFaces = mesh.EF[sharpEdges].ravel()
constVectors = np.repeat(edgeDirections/np.linalg.norm(edgeDirections, axis=1)[:, None], 2, axis=0)

#a raw field of just the constraints, for visualization
rawFieldConstraints = np.zeros((len(mesh.F), 3*N))
posInFace = np.zeros(len(mesh.F), dtype=int)
for face, vector in zip(constFaces, constVectors):
    rawFieldConstraints[face, 3*posInFace[face]:3*posInFace[face]+3] = vector
    posInFace[face] += 1
if N%2==0:
    rawFieldConstraints[:, 3*N//2:] = -rawFieldConstraints[:, :3*N//2]
constraintsField = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, N)
constraintsField.set_extrinsic_field(rawFieldConstraints)


def recompute_field():
    global rawFieldHard, rawFieldSoft
    pvData = dr.PolyVectorData(ftb, N)
    pvData.verbose = True
    pvData.constSpaces = constFaces
    pvData.constVectors = constVectors
    pvData.wSmooth = smoothWeight
    pvData.wRoSy = roSyWeight
    pvData.wAlignment = np.full(constFaces.size, -1.0)
    pvFieldHard = dr.polyvector_field(pvData)

    pvData.wAlignment = np.full(constFaces.size, alignWeight)
    pvFieldSoft = dr.polyvector_field(pvData)

    rawFieldHard = dr.polyvector_to_raw(pvFieldHard, N%2==0)
    dr.principal_matching(rawFieldHard)
    rawFieldSoft = dr.polyvector_to_raw(pvFieldSoft, N%2==0)
    dr.principal_matching(rawFieldSoft)


def update_visualization():
    viewer.set_cartesian_field(constraintsField, "Constraints", color=viewer.DEFAULT_VECTOR_CONSTRAINT_COLOR)
    viewer.highlight_faces(constFaces, "Constraint-face highlights")
    if viewingMode!=CONSTRAINTS:
        viewer.set_cartesian_field(rawFieldHard if viewingMode==HARD_PRESCRIPTION else rawFieldSoft, "PolyVector Field")
    if ps.has_point_cloud("PolyVector Field"):
        ps.get_point_cloud("PolyVector Field").set_enabled(viewingMode!=CONSTRAINTS)


def callbackFunc():
    global roSyWeight, alignWeight, viewingMode
    psim.PushItemWidth(300)
    changedRoSy, roSyWeight = psim.InputDouble("RoSy Weight", roSyWeight)
    changedAlign, alignWeight = psim.InputDouble("Alignment Weight", alignWeight)
    if changedRoSy or changedAlign:
        recompute_field()
        update_visualization()

    changed, viewingMode = psim.Combo("Viewing mode", viewingMode, ["Only Constraints", "Hard prescription", "Soft prescription"])
    if changed:
        update_visualization()

    if psim.Button("Save Raw Field"):
        dr.write_raw_field("polyvector.rawfield", rawFieldHard if viewingMode==HARD_PRESCRIPTION else rawFieldSoft)
    psim.PopItemWidth()


ps.init()
viewer.set_surface_mesh(mesh)
recompute_field()
update_visualization()
ps.set_user_callback(callbackFunc)
ps.show()
