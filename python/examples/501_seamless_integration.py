# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/501_SeamlessIntegration: integrates a field into a rotationally-seamless and a fully-seamless
# parameterization.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

meshWhole = dr.readOFF(str(TUTORIAL_DATA_PATH / "horsers.off"))
ftb = dr.PCFaceTangentBundle(meshWhole)
rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "horsers-cf.rawfield"), ftb)

#combing and cutting
curlNorm = dr.curl_matching(rawField)
print("curlNorm max:", curlNorm.max())

intData = dr.IntegrationData(rawField.N)
intData.verbose = True

print("Setting up Integration")
meshCut, combedField = dr.setup_integration(rawField, intData)
seams = np.unique(meshWhole.FE[intData.face2cut!=0])

intData.integralSeamless = False
print("Solving for permutationally-seamless integration")
cutUVRot, _ = dr.integrate(combedField, intData, meshCut)
#Extracting the UV from [U,V,-U, -V];
cutUVRot = cutUVRot[:, :2]
print("Done!")

intData.integralSeamless = True
print("Solving for integrally-seamless integration")
cutUVFull, _ = dr.integrate(combedField, intData, meshCut)
cutUVFull = cutUVFull[:, :2]
print("Done!")


def callbackFunc():
    psim.PushItemWidth(100)
    if psim.Button("Save Textured OBJ File"):
        dr.writeOBJ("horsers-param-rot-seamless.obj", meshCut, cutUVRot, meshCut.F, "squares.mtl", "material_0")
        dr.writeOBJ("horsers-param-full-seamless.obj", meshCut, cutUVFull, meshCut.F, "squares.mtl", "material_0")
    psim.PopItemWidth()


#viewer cut (texture) and whole (field) meshes
ps.init()
viewer.set_surface_mesh(meshWhole, "Original Mesh")
viewer.set_surface_mesh(meshCut, "Cut Mesh")
viewer.set_cartesian_field(combedField)
viewer.highlight_edges(meshWhole, seams, "Seams")
viewer.set_uv(cutUVRot, "Rotationally Seamless UV", "Cut Mesh")
viewer.set_uv(cutUVFull, "Full Seamless UV", "Cut Mesh")
ps.set_user_callback(callbackFunc)
ps.show()
