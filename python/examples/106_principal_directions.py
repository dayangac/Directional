# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/106_PrincipalDirections: the principal curvature directions and values at vertices.
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "botanic-garden-bubble.off"))

minField = np.hstack([mesh.minVertexPrincipalDirections, -mesh.minVertexPrincipalDirections])
maxField = np.hstack([mesh.maxVertexPrincipalDirections, -mesh.maxVertexPrincipalDirections])

ps.init()
psMesh = viewer.set_surface_mesh(mesh)
for name, rawField, color in (("Min Principal Direction", minField, (107.0/255.0, 8.0/255.0, 125.0/255.0)),
                              ("Max Principal Direction", maxField, (125.0/255.0, 107.0/255.0, 8.0/255.0)),
                              ("Vertex normals", mesh.vertexNormals, (8.0/255.0, 125.0/255.0, 107.0/255.0))):
    viewer.set_raw_field(mesh.V, rawField, name, 0.3*mesh.avgEdgeLength, color)

psMesh.add_scalar_quantity("Min curvature", mesh.vertexPrincipalCurvatures[:, 0], enabled=True)
psMesh.add_scalar_quantity("Max curvature", mesh.vertexPrincipalCurvatures[:, 1])
ps.show()
