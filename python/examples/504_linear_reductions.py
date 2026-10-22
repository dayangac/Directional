# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/504_LinearReductions: integrates a 6-field with sign symmetry, and with triangular symmetry.
from pathlib import Path

import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

SIGN_SYMMETRY, TRI_SYMMETRY = range(2)
viewingMode = SIGN_SYMMETRY

meshWhole = dr.readOFF(str(TUTORIAL_DATA_PATH / "dome.off"))
ftb = dr.PCFaceTangentBundle(meshWhole)
rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "dome-6.rawfield"), ftb)
N = rawField.N

#combing and cutting
dr.principal_matching(rawField)

intData = dr.IntegrationData(N)
print("Setting up Integration")
meshCut, combedField = dr.setup_integration(rawField, intData)

intData.verbose = False
intData.integralSeamless = True

print("Free (sign-symmetric) Integrating...")
NFunctionSign, _ = dr.integrate(combedField, intData, meshCut)
print("Done!")

print("Solving triangular-constrained integration...")
intData.set_triangular_symmetry(N)
meshCut, combedField = dr.setup_integration(rawField, intData)
NFunctionTri, _ = dr.integrate(combedField, intData, meshCut)
print("Done!")


def callbackFunc():
    global viewingMode
    psim.PushItemWidth(300)
    changed, viewingMode = psim.Combo("Viewing Mode", viewingMode, ["Sign Symmetry", "Triangular Symmetry"])
    if changed:
        viewer.set_isolines(meshCut, NFunctionSign if viewingMode==SIGN_SYMMETRY else NFunctionTri, rawField)
    psim.PopItemWidth()


ps.init()
viewer.set_surface_mesh(meshWhole)
viewer.set_cartesian_field(rawField)
viewer.set_isolines(meshCut, NFunctionSign, rawField)
ps.set_user_callback(callbackFunc)
ps.show()
