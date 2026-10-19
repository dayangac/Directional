# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/202_Sampling: a field with prescribed singularities, its principal matching, and its interpolation.
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

N = 2
globalRotation = 0.0
TRIVIAL_ONE_SING, TRIVIAL_PRINCIPAL_MATCHING, IMPLICIT_FIELD = range(3)
viewingMode = TRIVIAL_ONE_SING

mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
ftb = dr.PCFaceTangentBundle(mesh)
b = dr.readDMAT(str(TUTORIAL_DATA_PATH / "spheres_constFaces.dmat"))[:, 0].astype(int)

singVertices = np.array([35, 36])
singIndices = np.array([N, N])

field = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, N)


def update_directional_field():
    global field
    presSingIndices = np.zeros(len(ftb.cycleCurvatures), dtype=int)
    presSingIndices[singVertices] = singIndices
    dr.index_prescription(presSingIndices, N, globalRotation, field)

    if viewingMode==TRIVIAL_PRINCIPAL_MATCHING:
        dr.principal_matching(field)

    if viewingMode==IMPLICIT_FIELD:
        bc = field.extField[b, :3]
        bc = bc/np.linalg.norm(bc, axis=1)[:, None]
        powerField = dr.power_field(ftb, b, bc, np.full(b.size, -1.0), N)
        field = dr.power_to_raw(powerField, N, normalize=True)
        dr.principal_matching(field)

    viewer.set_cartesian_field(field)

    if viewingMode==TRIVIAL_ONE_SING:
        viewer.set_singularities(field, singElements=singVertices, singIndices=singIndices)


def callbackFunc():
    global globalRotation, viewingMode
    psim.PushItemWidth(100)
    psim.TextUnformatted("Prescribed singularity index: " + str(singIndices[0]))
    psim.SameLine()
    if psim.Button("+"):
        singIndices[0] += 1
        singIndices[1] -= 1
        update_directional_field()
    psim.SameLine()
    if psim.Button("-"):
        singIndices[0] -= 1
        singIndices[1] += 1
        update_directional_field()
    if psim.Button("Global Rotation"):
        globalRotation += np.pi/16.0
        update_directional_field()
    psim.PopItemWidth()

    changed, viewingMode = psim.Combo("Viewing Mode", viewingMode, ["Prescribed singularity", "Principal-matching singularities", "Interpolated Field"])
    if changed:
        update_directional_field()


ps.init()
viewer.set_surface_mesh(mesh)
viewer.highlight_faces(b)
ps.set_user_callback(callbackFunc)
update_directional_field()
ps.show()
