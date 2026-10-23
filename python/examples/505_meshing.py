# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/505_Meshing: meshes the integer isolines of seamless 4-, 7- and 11-functions into polygonal
# meshes (needs the module to be built with GMP).
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"


#port of the tutorial's polygonal_write_OFF.h
def polygonal_write_OFF(fileName, V, D, F):
    with open(fileName, "w") as file:
        file.write("OFF\n" + str(len(V)) + " " + str(len(D)) + " 0\n")
        np.savetxt(file, V, fmt="%.9g")
        for d, face in zip(D, F):
            file.write(" ".join(str(i) for i in [d, *face[:d]]) + "\n")


meshWhole = dr.readOFF(str(TUTORIAL_DATA_PATH / "vase.off"))
ftb = dr.PCFaceTangentBundle(meshWhole)
rawFields = [dr.read_raw_field(str(TUTORIAL_DATA_PATH / ("vase-" + str(N) + ".rawfield")), ftb) for N in (4, 7, 11)]

#combing and cutting
ps.init()
viewer.set_surface_mesh(meshWhole).set_enabled(False)
for i, rawField in enumerate(rawFields):
    N = rawField.N
    dr.principal_matching(rawField)

    intData = dr.IntegrationData(N)
    print("Setting up Integration #" + str(i))
    meshCut, combedField = dr.setup_integration(rawField, intData)

    intData.verbose = False
    intData.integralSeamless = True
    intData.roundSeams = False

    print("Solving integration for N=" + str(N))
    NFunction, _ = dr.integrate(combedField, intData, meshCut)
    print("Done!")

    #setting up mesh data from integration data
    mData = dr.setup_mesher(meshCut, intData)

    #meshing and saving
    mData.verbose = False
    print("Meshing N=" + str(N))
    VPolyMesh, DPolyMesh, FPolyMesh = dr.mesher(meshWhole, mData)
    print("Done!")

    psGlyphs = viewer.set_cartesian_field(combedField, str(N) + "-field")
    psIsolines = viewer.set_isolines(meshCut, NFunction, combedField, str(N) + "-function")

    #polyscope takes polygonal faces as lists of vertex indices
    name = "vase-" + str(N) + "-generated.off"
    polygonalMesh = ps.register_surface_mesh(name, VPolyMesh, [face[:d].tolist() for d, face in zip(DPolyMesh, FPolyMesh)])

    #Also saving file
    polygonal_write_OFF(name, VPolyMesh, DPolyMesh, FPolyMesh)
    if i!=0:
        psGlyphs.set_enabled(False)
        psIsolines.set_enabled(False)
        ps.get_point_cloud(str(N) + "-field singularities").set_enabled(False)
        polygonalMesh.set_enabled(False)

ps.show()
