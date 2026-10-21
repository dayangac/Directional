# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/604_HodgeBoundary: the Hodge decomposition of a 1-form on a mesh with boundaries.
from pathlib import Path

import numpy as np
import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "flatmesh.off"))

d0 = dr.d0_matrix(mesh)
d1 = dr.d1_matrix(mesh)
h1, invh1 = dr.hodge_star_1_matrix(mesh)
h2, invh2 = dr.hodge_star_2_matrix(mesh)

#demonstrating the exact sequences even with boundary conditions
print("exact sequence identity (should be exactly zero):", abs(d1@d0).max())

bettiNumber = len(mesh.EV)-(len(mesh.V)-1)-len(mesh.F)
harmBasis = dr.cohomology_basis(d0, d1, h1, bettiNumber)
print("Betti number:", bettiNumber)

COM = np.mean(mesh.V, axis=0)
z0GT = 3.0*np.cos((mesh.V[:, 0]-COM[1])/4.0)*np.cos((mesh.V[:, 2]-COM[2])/4.0)
curlGT = mesh.faceAreas*np.sum(np.sin((mesh.barycenters-COM)/2.0), axis=1)
curlGT -= np.mean(curlGT)  #Due to Neumann boundary conditions (tangent coexact field), vector potential function adds up to zero

#Generating an artificial composition and then reproducing it through decomposition.
harmGT = harmBasis[:, 0]
z1ExactGT = d0@z0GT
z1CoexactGT, curlFiltered = dr.project_exact(d1, h1, curlGT, gaugeFixing=True)
#creating balanced GT results for exact, coexact, and harmonic
z1ExactGT /= np.sqrt(z1ExactGT@h1@z1ExactGT)
z1CoexactGT /= np.sqrt(z1CoexactGT@h1@z1CoexactGT)
z1 = harmGT+z1ExactGT+z1CoexactGT

z1Exact, z1Coexact, z1Harmonic, z0, z2 = dr.hodge_decomposition(d0, d1, h1, h2, z1, 1)
print("Exact reproduction:", np.max(np.abs(z1Exact-z1ExactGT)))
print("Coexact reproduction:", np.max(np.abs(z1Coexact-z1CoexactGT)))
print("Harmonic reproduction:", np.max(np.abs(z1Harmonic-harmGT)))

ps.init()
psMesh = viewer.set_surface_mesh(mesh)
psMesh.add_scalar_quantity("0-form potential", z0, defined_on="vertices", enabled=True)
viewer.set_1form(mesh, z1, "Original field", 5.0)
viewer.set_1form(mesh, z1Exact, "Exact field", 5.0)
psMesh.add_scalar_quantity("dual 0-form potential", z2/mesh.faceAreas, defined_on="faces", enabled=True)
viewer.set_1form(mesh, z1Coexact, "Coexact field", 5.0)
viewer.set_1form(mesh, z1Harmonic, "Harmonic field", 5.0)
ps.show()
