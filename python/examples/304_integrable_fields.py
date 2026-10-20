# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/304_IntegrableFields: a curl-free PolyVector field, by iterating soft RoSy and curl projection.
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 4
smoothWeight, roSyWeight, alignWeight = 1.0, 5.0, 1.0

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "cheburashka.off"))
ftb = dr.PCFaceTangentBundle(mesh)

#constraining both faces of every sharp edge to the edge direction
sharpEdges = np.flatnonzero(np.sum(mesh.faceNormals[mesh.EF[:, 0]]*mesh.faceNormals[mesh.EF[:, 1]], axis=1)<0.1)
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
pvData.wRoSy = roSyWeight

pvFieldOrig = dr.polyvector_field(pvData)
rawFieldOrig = dr.polyvector_to_raw(pvFieldOrig, N%2==0)
curlOrig = dr.curl_matching(rawFieldOrig)

#alternating an implicit smoothing step, a soft RoSy step, and a projection to the nearest curl-free field
numIterations = 10
pvData.iterationMode = True
pvFieldCurlFree = dr.polyvector_field(pvData)
dr.polyvector_iterate(pvData, pvFieldCurlFree, [dr.soft_rosy, dr.curl_projection], numIterations)
rawFieldCurlFree = dr.polyvector_to_raw(pvFieldCurlFree, N%2==0)
curlCF = dr.curl_matching(rawFieldCurlFree)
print("Maximum final curl:", np.max(np.abs(curlCF)))

ps.init()
psMesh = viewer.set_surface_mesh(mesh)
viewer.set_raw_field(constSources, constVectors, "Constraints", mesh.avgEdgeLength, viewer.DEFAULT_VECTOR_CONSTRAINT_COLOR)
viewer.highlight_faces(constFaces, "Const Faces")
viewer.set_cartesian_field(rawFieldOrig, "Original Field")
viewer.set_cartesian_field(rawFieldCurlFree, "Curl-free Field", color=(107.0/255.0, 8.0/255.0, 125.0/255.0))
curlRange = (np.min(np.abs(curlOrig)), np.max(np.abs(curlOrig))/10.0)
psMesh.add_scalar_quantity("Original Abs Curl", curlOrig, defined_on="edges", enabled=True, vminmax=curlRange)
psMesh.add_scalar_quantity("Optimized Abs Curl", curlCF, defined_on="edges", enabled=True, vminmax=curlRange)
ps.show()
