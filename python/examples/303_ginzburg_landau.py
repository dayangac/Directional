# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/303_GinzburgLandau: a unit RoSy field, by iterating implicit smoothing and projection to unit RoSy.
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 4
smoothWeight, alignWeight = 1.0, 1.0

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "deckel_input.off"))
ftb = dr.PCFaceTangentBundle(mesh)

#constraining both faces of every sharp edge to the edge direction
sharpEdges = np.flatnonzero(np.sum(mesh.faceNormals[mesh.EF[:, 0]]*mesh.faceNormals[mesh.EF[:, 1]], axis=1)<0.5)
edgeDirections = mesh.V[mesh.EV[sharpEdges, 0]]-mesh.V[mesh.EV[sharpEdges, 1]]
constFaces = mesh.EF[sharpEdges].ravel()
constVectors = np.repeat(edgeDirections/np.linalg.norm(edgeDirections, axis=1)[:, None], 2, axis=0)
constSources = mesh.barycenters[constFaces]  #For visualization

pvData = dr.PolyVectorData(ftb, N)
pvData.verbose = True
pvData.constSpaces = constFaces
pvData.constVectors = constVectors
pvData.wAlignment = np.full(constFaces.size, alignWeight)
pvData.wSmooth = smoothWeight
pvData.wRoSy = -1.0  #Perfect RoSy
pvData.initImplicitFactor = 0.2
pvData.implicitScheduler = 1.0

pvFieldOrig = dr.polyvector_field(pvData)
rawFieldOrig = dr.polyvector_to_raw(pvFieldOrig, N%2==0)
dr.principal_matching(rawFieldOrig)

#Ginzburg-Landau: alternating an implicit smoothing step and a projection to unit RoSy
pvData.iterationMode = True
numIterations = 100
pvFieldGL = dr.polyvector_field(pvData)
dr.polyvector_iterate(pvData, pvFieldGL, [dr.hard_rosy], numIterations)
rawFieldGL = dr.polyvector_to_raw(pvFieldGL, N%2==0)
dr.principal_matching(rawFieldGL)

ps.init()
viewer.set_surface_mesh(mesh)
viewer.set_raw_field(constSources, constVectors, "Constraints", mesh.avgEdgeLength, viewer.DEFAULT_VECTOR_CONSTRAINT_COLOR)
viewer.highlight_faces(constFaces, "Const Faces")
viewer.set_cartesian_field(rawFieldOrig, "Original Field")
viewer.set_cartesian_field(rawFieldGL, "Ginzburg-Landau Field", color=(107.0/255.0, 8.0/255.0, 125.0/255.0))
ps.show()
