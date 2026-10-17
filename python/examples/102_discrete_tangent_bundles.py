# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/102_DiscreteTangentBundles: the same constraint on a face-based and on a vertex-based field.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 2
viewFaceField = True

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "elephant.obj"))
ftb = dr.PCFaceTangentBundle(mesh)
vtb = dr.IntrinsicVertexTangentBundle(mesh)


def callbackFunc():
    global viewFaceField
    if psim.Button("Toggle Vertex Field" if viewFaceField else "Toggle Face Field"):
        viewFaceField = not viewFaceField
    for name, enabled in (("Face-Based Field", viewFaceField), ("Vertex-Based Field", not viewFaceField)):
        ps.get_point_cloud(name).set_enabled(enabled)
        ps.get_point_cloud(name + " singularities").set_enabled(enabled)


constFaces = np.array([0])
constVertices = np.array([mesh.F[0, 1]])
constVectors = (mesh.V[mesh.F[0, 2]]-mesh.V[mesh.F[0, 1]]).reshape(1, 3)

powerVertexField = dr.power_field(vtb, constVertices, constVectors, np.full(constVertices.size, -1.0), N)
powerFaceField = dr.power_field(ftb, constFaces, constVectors, np.full(constFaces.size, -1.0), N)
rawFaceField = dr.power_to_raw(powerFaceField, N, normalize=True)
rawVertexField = dr.power_to_raw(powerVertexField, N, normalize=True)
dr.principal_matching(rawFaceField)
dr.principal_matching(rawVertexField)

ps.init()
viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(rawFaceField, "Face-Based Field")
viewer.set_cartesian_field(rawVertexField, "Vertex-Based Field")
ps.set_user_callback(callbackFunc)
ps.show()
