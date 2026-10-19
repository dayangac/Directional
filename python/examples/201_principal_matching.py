# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/201_PrincipalMatching: click on a face to see its field combed with its neighbors by the matching.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
ftb = dr.PCFaceTangentBundle(mesh)
field = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), ftb)
dr.principal_matching(field)
N = field.N


def callbackFunc():
    if psim.IsMouseClicked(0) and not psim.GetIO().WantCaptureMouse:
        pickResult = ps.pick(screen_coords=psim.GetMousePos())
        if pickResult.is_hit and pickResult.structure_name=="Mesh" and "bary_coords" in pickResult.structure_data:
            currF = pickResult.structure_data["index"]
            print("clicked face", currF)

            #Polyscope doesn't allow different color vectors, so we are setting two fields - one combed one not
            sourcesCombed = [mesh.barycenters[currF]+0.001*mesh.faceNormals[currF]]
            rawFieldCombed = [field.extField[currF]]
            for edge in mesh.FE[currF]:
                otherFace = (mesh.EF[edge, 1] if mesh.EF[edge, 0]==currF else mesh.EF[edge, 0])
                if otherFace==-1:  #boundary edge
                    continue
                zeroInFace = (field.matching[edge] if mesh.EF[edge, 0]==currF else -field.matching[edge])%N
                #reordering the vectors in the rawfield
                sourcesCombed.append(mesh.barycenters[otherFace]+0.001*mesh.faceNormals[otherFace])  #to avoid z-fighting with the uncombed field
                rawFieldCombed.append(np.roll(field.extField[otherFace], -3*zeroInFace))

            #Visualizing a local field just of combed directionals
            viewer.set_raw_field(np.array(sourcesCombed), np.array(rawFieldCombed), "Combed Field", 0.3*mesh.avgEdgeLength,
                                 [viewer.indexed_glyph_colors(i, N, signSymmetry=False) for i in range(N)])


ps.init()
psMesh = viewer.set_surface_mesh(mesh)
psMesh.set_selection_mode("faces_only")
viewer.set_cartesian_field(field)
ps.set_user_callback(callbackFunc)
ps.show()
