# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/401_IndexPrescription: a field with prescribed indices on local, boundary and generator cycles.
# Click on a vertex to choose its cycle.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 2
currCycle = 0
globalRotation = 0.0

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "fertility.obj"))
ftb = dr.PCFaceTangentBundle(mesh)
field = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, N)
numCycles = ftb.cycles.shape[0]

dr.read_singularities(str(TUTORIAL_DATA_PATH / "fertility.sings"), field)
cycleIndices = np.zeros(numCycles, dtype=int)
cycleIndices[ftb.local2Cycle[field.singLocalCycles]] = field.singIndices

#the faces of the edges of each cycle, for highlighting
cycleEntries = ftb.cycles.tocoo()
cycleFaces = [[] for i in range(numCycles)]
for cycle, edge in zip(cycleEntries.row, mesh.innerEdges[cycleEntries.col]):
    cycleFaces[cycle] += [f for f in mesh.EF[edge] if f!=-1]


def update_field():
    total = np.sum(cycleIndices[:numCycles-mesh.numGenerators])
    if mesh.eulerChar*N!=total:
        print("Warning: All non-generator singularities should add up to N * the Euler characteristic.")
        print("Total indices: " + str(total) + "/" + str(N))
        print("Expected: " + str(mesh.eulerChar*N) + "/" + str(N))

    _, linfError = dr.index_prescription(cycleIndices, N, globalRotation, field)
    print("Index prescription linfError:", linfError)
    viewer.set_cartesian_field(field)
    viewer.highlight_faces(cycleFaces[currCycle])


def update_singularities():
    singVertices = np.flatnonzero(cycleIndices[ftb.local2Cycle])
    viewer.set_singularities(field, singElements=singVertices, singIndices=cycleIndices[ftb.local2Cycle[singVertices]])


def callbackFunc():
    global currCycle, globalRotation
    psim.PushItemWidth(100)
    psim.TextUnformatted("Prescribed cycle index: " + str(cycleIndices[currCycle]))
    psim.SameLine()
    if psim.Button("+"):
        cycleIndices[currCycle] += 1
        update_field()
        update_singularities()
    psim.SameLine()
    if psim.Button("-"):
        cycleIndices[currCycle] -= 1
        update_field()
        update_singularities()
    if psim.Button("Global Rotation"):
        globalRotation += np.pi/10.0
        update_field()
        update_singularities()

    if psim.Button("Change boundary cycle") and len(mesh.boundaryLoops)>0:
        firstBoundaryCycle = numCycles-len(mesh.boundaryLoops)-mesh.numGenerators
        currCycle = currCycle+1 if firstBoundaryCycle<=currCycle<numCycles-mesh.numGenerators-1 else firstBoundaryCycle
        viewer.highlight_faces(cycleFaces[currCycle])

    if psim.Button("Change generator cycle") and mesh.numGenerators>0:
        firstGeneratorCycle = numCycles-mesh.numGenerators
        currCycle = currCycle+1 if firstGeneratorCycle<=currCycle<numCycles-1 else firstGeneratorCycle
        print("Current generator cycle:", currCycle)
        viewer.highlight_faces(cycleFaces[currCycle])

    if psim.Button("Save field"):
        dr.write_raw_field("index-prescription.rawfield", field)
        print("Saved raw field")
    psim.PopItemWidth()

    if psim.IsMouseClicked(0) and not psim.GetIO().WantCaptureMouse:
        pickResult = ps.pick(screen_coords=psim.GetMousePos())
        if pickResult.is_hit and pickResult.structure_name=="Mesh" and pickResult.structure_data["element_type"]=="vertex":
            currVertex = pickResult.structure_data["index"]
            print("clicked vertex", currVertex)
            currCycle = ftb.local2Cycle[currVertex]
            viewer.highlight_faces(cycleFaces[currCycle])


ps.init()
psMesh = viewer.set_surface_mesh(mesh)
psMesh.set_selection_mode("vertices_only")
viewer.set_cartesian_field(field)
viewer.highlight_faces(cycleFaces[currCycle])
update_field()
update_singularities()
ps.set_user_callback(callbackFunc)
ps.show()
