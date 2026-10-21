# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/601_FaceBasedFEM: conforming and non-conforming gradients, and the exact sequences they form.
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "60745__sf.obj"))

confVec = 50.0*np.sin(mesh.V[:, 2]/20.0)
nonConfVec = np.linalg.norm(mesh.midEdges-mesh.midEdges[17160], axis=1)

Gv = dr.conf_gradient_matrix_2D(mesh)
Ge = dr.non_conf_gradient_matrix_2D(mesh)
J = dr.face_vector_rotation_matrix_2D(mesh)
C = dr.curl_matrix_2D(mesh)
D = dr.div_matrix_2D(mesh)

#demonstrating the exact sequences
print("max abs curl of exact field (should be numerically zero):", np.max(np.abs(C@Gv@confVec)))
print("max abs divergence of coexact field (should be numerically zero):", np.max(np.abs(D@J@Ge@nonConfVec)))

gradRawField = (Gv@confVec).reshape(-1, 3)
rotCogradRawField = (J@Ge@nonConfVec).reshape(-1, 3)

ps.init()
psMesh = viewer.set_surface_mesh(mesh)
psMesh.add_scalar_quantity("Non-Conforming Function", nonConfVec, defined_on="edges", enabled=True)
psMesh.add_scalar_quantity("Conforming Function", confVec, defined_on="vertices", enabled=True)
viewer.set_raw_field(mesh.barycenters, gradRawField, "Gradient field", 1.0*mesh.avgEdgeLength)
viewer.set_raw_field(mesh.barycenters, rotCogradRawField, "Rot. Cogradient field", 1.0*mesh.avgEdgeLength).set_enabled(False)
ps.show()
