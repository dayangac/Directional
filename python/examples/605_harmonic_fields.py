# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/605_HarmonicFields: a basis of the harmonic vector fields of a high-genus surface.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

currHarmBasis = 0

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "129110__sf.obj"))
ftb = dr.PCFaceTangentBundle(mesh)

#Must use intrinsic since otherwise the harmonic field will have spurious normal components
G = dr.conf_gradient_matrix_2D(mesh, True)
C = dr.curl_matrix_2D(mesh, True)
#the mass matrix of intrinsic (2D) vectors; the C++ tutorial uses the default of 3D vectors, which does not match G
Mx = dr.face_mass_matrix_2D(mesh, False, 2)
IE = dr.face_intrinsic_to_extrinsic_matrix_2D(mesh)

bettiNumber = len(mesh.EV)-(len(mesh.V)-1)-(len(mesh.F)-1)
print("Computing cohomology basis...")
harmBasis = dr.cohomology_basis(G, C, Mx, bettiNumber)
print("bettiNumber:", bettiNumber)
print("divergence of harmonic basis:", np.max(np.abs(G.T@Mx@harmBasis)))
print("curl of harmonic basis:", np.max(np.abs(C@harmBasis)))

harmField = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, 1)


def update_field():
    harmField.set_extrinsic_field((IE@harmBasis[:, currHarmBasis]).reshape(-1, 3))
    viewer.set_cartesian_field(harmField, "Harmonic Field", 50.0)


def callbackFunc():
    global currHarmBasis
    psim.PushItemWidth(100)
    psim.TextUnformatted("Harmonic basis function: " + str(currHarmBasis))
    psim.SameLine()
    if psim.Button("+"):
        currHarmBasis = (currHarmBasis+1)%harmBasis.shape[1]
        update_field()
    psim.SameLine()
    if psim.Button("-"):
        currHarmBasis = (currHarmBasis+harmBasis.shape[1]-1)%harmBasis.shape[1]
        update_field()
    psim.PopItemWidth()


ps.init()
ps.set_user_callback(callbackFunc)
viewer.set_surface_mesh(mesh)
update_field()
ps.show()
