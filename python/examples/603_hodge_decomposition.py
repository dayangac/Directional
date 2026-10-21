# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/603_HodgeDecomposition: decomposing a face-based vector field into gradient, rotated cogradient and harmonic parts.
from pathlib import Path

import numpy as np
import polyscope as ps
import scipy.sparse

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "1146164.off"))

#Must use intrinsic since otherwise the harmonic field will have spurious normal components
G = dr.conf_gradient_matrix_2D(mesh, True)
C = dr.curl_matrix_2D(mesh, True)
#Face-based mass matrix for vectors
Mx = dr.face_mass_matrix_2D(mesh, False, 2)
#Inverse mass matrix
iMx = dr.face_mass_matrix_2D(mesh, True, 2)
#Conversion matrix intrinsic to extrinsic
IE = dr.face_intrinsic_to_extrinsic_matrix_2D(mesh)

bettiNumber = len(mesh.EV)-(len(mesh.V)-1)-(len(mesh.F)-1)
print("Extracting the basis of harmonic fields...")
harmBasis = dr.cohomology_basis(G, C, Mx, bettiNumber)
harmGT = harmBasis[:, 0]

vertexVec = np.linalg.norm(mesh.V-mesh.V[296], axis=1)
midEdgeVec = np.sum(np.sin(mesh.midEdges/2.0), axis=1)

#Doing an artifical composition and decomposition. For results to be meaningful, the fields must be of more or less the same magnitude, so normalizing them.
gradientGT = G@vertexVec
gradientGT /= np.sqrt(gradientGT@Mx@gradientGT)
rotCogradientGT = -iMx@C.T@midEdgeVec  #This is equivalent to J*Ge*midEdgeVec
rotCogradientGT /= np.sqrt(rotCogradientGT@Mx@rotCogradientGT)
origFieldVec = harmGT+gradientGT+rotCogradientGT

#Using identity matrix since the convention is that Hodge decomposition in face based fields uses the non-conforming mid-edge based function
I = scipy.sparse.identity(len(mesh.EV), format="csc")
gradFieldVec, rotCogradFieldVec, harmFieldVec, vertexFunction, edgeFunction = dr.hodge_decomposition(G, C, Mx, I, origFieldVec, 1)
print("Reproduction of exact part (numerically zero):", np.max(np.abs(gradFieldVec-gradientGT)))
print("Reproduction of coexact part (numerically zero):", np.max(np.abs(rotCogradFieldVec-rotCogradientGT)))
print("Reproduction of harmonic part (numerically zero):", np.max(np.abs(harmFieldVec-harmGT)))

ps.init()
psMesh = viewer.set_surface_mesh(mesh)
psMesh.add_scalar_quantity("Vertex Function", vertexFunction, defined_on="vertices")
psMesh.add_scalar_quantity("Edge Function", edgeFunction, defined_on="edges")
#Visualization of the extrinsic fields
for name, fieldVec, enabled in (("Original Field", origFieldVec, True), ("Gradient Field", gradFieldVec, False),
                                ("Rotated Cogradient Field", rotCogradFieldVec, False), ("Harmonic Field", harmFieldVec, False)):
    viewer.set_raw_field(mesh.barycenters, (IE@fieldVec).reshape(-1, 3), name, 20.0*mesh.avgEdgeLength).set_enabled(enabled)
ps.show()
