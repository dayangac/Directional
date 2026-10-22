# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/105_Sparsity: shows the glyphs of a field on a sparse sample of its tangent spaces.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 2
sparsity = 0


def callbackFunc():
    global sparsity
    psim.PushItemWidth(100)
    psim.Text("Sparsity: " + str(sparsity))
    psim.SameLine()
    if psim.Button("+"):
        sparsity += 1
        viewer.set_cartesian_field(field, unitToAvgLengthRatio=0.3*(sparsity+1.0), sparsity=sparsity)
    psim.SameLine()
    if psim.Button("-"):
        sparsity = max(sparsity-1, 0)
        viewer.set_cartesian_field(field, unitToAvgLengthRatio=0.3*(sparsity+1.0), sparsity=sparsity)
    psim.PopItemWidth()


mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "armadillo.obj"))
ftb = dr.PCFaceTangentBundle(mesh)
constFaces = np.array([0])
constVectors = (mesh.V[mesh.F[0, 1]]-mesh.V[mesh.F[0, 2]])[None, :]
powerField = dr.power_field(ftb, constFaces, constVectors, np.full(len(constFaces), -1.0), N)
field = dr.power_to_raw(powerField, N, True)

ps.init()
ps.set_user_callback(callbackFunc)
viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(field)
ps.show()
