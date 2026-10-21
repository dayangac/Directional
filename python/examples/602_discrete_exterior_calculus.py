# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/602_DiscreteExteriorCalculus: the coexact 1-form of a given curl, with the diagonal and the Whitney metrics.
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "697224__sf.obj"))

#giving a 2-form of constant (pointwise curl).
z2 = mesh.faceAreas*np.sin(mesh.barycenters[:, 0]/40.0)*np.cos(mesh.barycenters[:, 1]/40.0)
#making it a physical curl quantity
z2 -= np.mean(z2)

d0 = dr.d0_matrix(mesh)
d1 = dr.d1_matrix(mesh)
hodgeStar, invHodgeStar = dr.hodge_star_1_matrix(mesh)
M1 = dr.linear_whitney_mass_matrix(mesh)

L1 = d0.T@M1@d0
L2 = d0.T@hodgeStar@d0
print("Exact laplacian identity (should be numerically zero):", abs(L1-L2).max())

z1Diag, z2Exact = dr.project_exact(d1, hodgeStar, z2, gaugeFixing=True)
print("Reproducing the original curl (z2Diag, should be numerically zero):", np.max(np.abs(z2-z2Exact)))
z1, z2Exact = dr.project_exact(d1, M1, z2, gaugeFixing=True)
print("Reproducing the original curl (z2, should be numerically zero):", np.max(np.abs(z2-z2Exact)))
print("Difference between z1 and z1Diag (small, not zero):", np.max(np.abs(z1-z1Diag)))

ps.init()
psMesh = viewer.set_surface_mesh(mesh)
psMesh.add_scalar_quantity("Integrated face curl", z2, defined_on="faces", enabled=True)
viewer.set_1form(mesh, z1, "Coexact field", 1.0, 2, 0.2)
viewer.set_1form(mesh, z1Diag, "Coexact field diag", 1.0, 2, 0.2)
ps.show()
