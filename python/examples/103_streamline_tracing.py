# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/103_StreamlineTracing: traces streamlines of a 3-field while the viewer runs.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 3
advanceTracing = False

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "lion.off"))
ftb = dr.PCFaceTangentBundle(mesh)


def advance_streamlines(dTimetoEdgeLengthRatio):
    dr.streamlines_next(slData, slState, dTimetoEdgeLengthRatio*ftb.avgAdjLength)
    viewer.set_streamlines(slState, width=0.05*ftb.avgAdjLength)


def callbackFunc():
    global advanceTracing
    if advanceTracing:
        advance_streamlines(0.5)
    if psim.Button("stop tracing"):
        advanceTracing = False
    elif psim.Button("resume tracing"):
        advanceTracing = True


constFaces = np.array([0])
constVectors = mesh.V[mesh.F[0, 1]]-mesh.V[mesh.F[0, 0]]
constVectors = (constVectors/np.linalg.norm(constVectors)).reshape(1, 3)
alignWeights = np.array([-1.0])

powerField = dr.power_field(ftb, constFaces, constVectors, alignWeights, N)
field = dr.power_to_raw(powerField, N, normalize=True)

slData, slState = dr.StreamlineData(), dr.StreamlineState()
dr.streamlines_init(field, np.empty(0, dtype=int), 3.0, slData, slState)

ps.init()
viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(field)
advance_streamlines(0.5)  #to get the initial step
ps.set_user_callback(callbackFunc)
ps.show()
