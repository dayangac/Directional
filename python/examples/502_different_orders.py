# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/502_DifferentOrders: seamless integration of fields of orders 2, 4, 7 and 11.
from pathlib import Path

import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

meshWhole = dr.readOFF(str(TUTORIAL_DATA_PATH / "vase.off"))
ftb = dr.PCFaceTangentBundle(meshWhole)
rawFields = [dr.read_raw_field(str(TUTORIAL_DATA_PATH / ("vase-" + str(N) + ".rawfield")), ftb) for N in (2, 4, 7, 11)]

#combing and cutting
ps.init()
viewer.set_surface_mesh(meshWhole)
for i, rawField in enumerate(rawFields):
    N = rawField.N
    dr.principal_matching(rawField)

    intData = dr.IntegrationData(N)
    print("Setting up Integration N=" + str(N))
    meshCut, combedField = dr.setup_integration(rawField, intData)

    intData.verbose = False
    intData.integralSeamless = True
    intData.roundSeams = False

    print("Solving integration N=" + str(N))
    NFunction, _ = dr.integrate(combedField, intData, meshCut)

    print("Done!")
    psGlyphs = viewer.set_cartesian_field(combedField, str(N) + "-field")
    psIsolines = viewer.set_isolines(meshCut, NFunction, combedField, str(N) + "-function", 0.05)
    if i!=0:
        psGlyphs.set_enabled(False)
        ps.get_point_cloud(str(N) + "-field singularities").set_enabled(False)
        psIsolines.set_enabled(False)

ps.show()
