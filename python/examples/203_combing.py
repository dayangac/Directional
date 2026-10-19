# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/203_Combing: combs a field so that its matching is trivial everywhere but on a set of seams.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

showCombed = False

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
ftb = dr.PCFaceTangentBundle(mesh)
origField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), ftb)

#computing the principal matching and then combing the field to trivialize the matching anywhere but a set of seams
dr.principal_matching(origField)
combedField = dr.combing(origField)
combedColors = [viewer.indexed_glyph_colors(i, origField.N, signSymmetry=False) for i in range(origField.N)]


def callbackFunc():
    global showCombed
    psim.PushItemWidth(100)
    if psim.Button("Show Original Field" if showCombed else "Show Combed Field"):
        showCombed = not showCombed
        viewer.set_cartesian_field(combedField if showCombed else origField, color=combedColors)
    psim.PopItemWidth()


ps.init()
psMesh = viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(origField, color=combedColors)
viewer.highlight_edges(mesh, np.flatnonzero(combedField.matching!=0), "Combing seams")
psMesh.add_scalar_quantity("Matching effort", origField.effort, defined_on="edges", enabled=True)
ps.set_user_callback(callbackFunc)
ps.show()
