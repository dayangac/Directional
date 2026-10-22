# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/503_SeamsSingsRounding: integer-seamless integration by rounding either the seams or the
# singularities.
from pathlib import Path

import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

SEAMS_ROUNDING, SINGS_ROUNDING = range(2)
viewingMode = SEAMS_ROUNDING

meshWhole = dr.readOFF(str(TUTORIAL_DATA_PATH / "train-station.off"))
ftb = dr.PCFaceTangentBundle(meshWhole)
rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "train-station-5.rawfield"), ftb)

#combing and cutting
dr.principal_matching(rawField)

intData = dr.IntegrationData(rawField.N)
print("Setting up Integration")
meshCut, combedField = dr.setup_integration(rawField, intData)

intData.verbose = False
intData.integralSeamless = True
intData.roundSeams = True

print("Seams-rounding Integrating...")
NFunctionSeams, _ = dr.integrate(combedField, intData, meshCut)
print("Done!")

intData.roundSeams = False
meshCut, combedField = dr.setup_integration(rawField, intData)
print("Singularity-rounding integration...")
NFunctionSings, _ = dr.integrate(combedField, intData, meshCut)
print("Done!")


def callbackFunc():
    global viewingMode
    psim.PushItemWidth(100)
    changed, viewingMode = psim.Combo("Viewing Mode", viewingMode, ["Rounding seams", "Rounding Singularities"])
    if changed:
        viewer.set_isolines(meshCut, NFunctionSeams if viewingMode==SEAMS_ROUNDING else NFunctionSings, rawField)
    psim.PopItemWidth()


ps.init()
viewer.set_surface_mesh(meshWhole)
viewer.set_cartesian_field(rawField)
viewer.set_isolines(meshCut, NFunctionSeams, rawField)
ps.set_user_callback(callbackFunc)
ps.show()
