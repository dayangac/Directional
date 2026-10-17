# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/301_PowerFields: click on the mesh to constrain the nearest vertex.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 4
normalizeField = False
viewFieldHard = True

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "rocker-arm2500.obj"))
vtb = dr.IntrinsicVertexTangentBundle(mesh)
constVertices = np.empty(0, dtype=int)
constVectors = np.empty((0, 3))
alignWeights = np.empty(0)


def recompute_field():
    global powerFieldHard, powerFieldSoft
    powerFieldHard = dr.power_field(vtb, constVertices, constVectors, np.full(constVertices.size, -1.0), N, normalizeField)
    powerFieldSoft = dr.power_field(vtb, constVertices, constVectors, alignWeights, N, normalizeField)


def update_visualization():
    rawField = dr.power_to_raw(powerFieldHard if viewFieldHard else powerFieldSoft, N)
    dr.principal_matching(rawField)

    viewer.set_cartesian_field(rawField, "Power field")
    if constVertices.size>0:
        viewer.set_raw_field(mesh.V[constVertices], constVectors, "Constraints", 0.3*vtb.avgAdjLength, viewer.DEFAULT_VECTOR_CONSTRAINT_COLOR)


def callbackFunc():
    global normalizeField, viewFieldHard, constVertices, constVectors, alignWeights

    changed, normalizeField = psim.Checkbox("Normalize Field", normalizeField)
    if changed:
        recompute_field()
        update_visualization()

    changed, currentItem = psim.Combo("Viewing mode", 0 if viewFieldHard else 1, ["Hard prescription", "Soft prescription"])
    if changed:
        viewFieldHard = (currentItem==0)
        update_visualization()

    if psim.IsMouseClicked(0) and not psim.GetIO().WantCaptureMouse:
        pickResult = ps.pick(screen_coords=psim.GetMousePos())
        if pickResult.is_hit and pickResult.structure_name=="Mesh" and "bary_coords" in pickResult.structure_data:
            currF = pickResult.structure_data["index"]
            currVertex = int(mesh.F[currF, np.argmax(pickResult.structure_data["bary_coords"])])

            i = np.flatnonzero(constVertices==currVertex)
            if i.size==0:
                constVertices = np.append(constVertices, currVertex)
                constVectors = np.vstack([constVectors, np.zeros(3)])
                alignWeights = np.append(alignWeights, 10e-4 if alignWeights.size==0 else alignWeights[-1])
                i = constVertices.size-1

            constVector = pickResult.position-mesh.V[currVertex]
            constVectors[i] = constVector/np.linalg.norm(constVector)

            recompute_field()
            update_visualization()


ps.init()
psMesh = viewer.set_surface_mesh(mesh)
psMesh.set_selection_mode("faces_only")
ps.set_user_callback(callbackFunc)
recompute_field()
update_visualization()
ps.show()
