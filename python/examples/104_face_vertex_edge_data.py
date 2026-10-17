# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/104_FaceVertexEdgeData: scalar data on faces, vertices and edges, next to a field.
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "eight.off"))
ftb = dr.PCFaceTangentBundle(mesh)
field = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "eight.rawfield"), ftb)

#Face data - the x component of the face normals
faceData = mesh.faceNormals[:, 0]

#vertex data: sin(z coordinate)
vertexData = np.sin(10.0*mesh.V[:, 2])

#Edge data - the (squared) effort of the field (under principal matching)
dr.principal_matching(field)
edgeData = field.effort**2

ps.init()
psMesh = viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(field)
psMesh.add_scalar_quantity("x of normal", faceData, defined_on="faces", enabled=True)
psMesh.add_scalar_quantity("sin(y)", vertexData, defined_on="vertices", enabled=True)
psMesh.add_scalar_quantity("principal effort", edgeData, defined_on="edges", enabled=True)
ps.show()
