# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Polyscope counterparts of the DirectionalViewer methods used by the tutorials, with the same colors and sizes.
import numpy as np
import polyscope as ps

DEFAULT_FACE_COLOR = (243.0/255.0, 241.0/255.0, 216.0/255.0)
DEFAULT_GLYPH_COLOR = (0.05, 0.05, 1.0)
#singularity colors for indices -3..3 (the index 0 color is the face color); lower is black, higher is white
INDEX_COLORS = {-3: (1.0, 1.0, 0.0), -2: (1.0, 0.5, 0.0), -1: (1.0, 0.0, 0.0), 0: DEFAULT_FACE_COLOR,
                1: (0.0, 1.0, 0.0), 2: (0.0, 1.0, 0.5), 3: (0.0, 1.0, 1.0)}


def default_index_color(index):
    if index<=-4:
        return (0.0, 0.0, 0.0)
    if index>=4:
        return (1.0, 1.0, 1.0)
    return INDEX_COLORS[index]


def set_surface_mesh(mesh, name="Mesh"):
    psMesh = ps.register_surface_mesh(name, mesh.V, mesh.F, color=DEFAULT_FACE_COLOR)
    #Directional's edge order already matches polyscope's, which makes edge data (such as effort) displayable
    psMesh.set_edge_permutation(np.arange(len(mesh.EV)))
    return psMesh


def set_raw_field(sources, rawField, name="Field", unitLength=1.0, color=DEFAULT_GLYPH_COLOR):
    psGlyphs = ps.register_point_cloud(name, sources, radius=10e-6, point_render_mode="quad")
    for i in range(rawField.shape[1]//3):
        psGlyphs.add_vector_quantity("Vector " + str(i), rawField[:, 3*i:3*i+3]*unitLength, enabled=True,
                                     vectortype="ambient", color=color)
    return psGlyphs


def set_singularities(field, name="Field", radiusRatio=0.3):
    psSings = ps.register_point_cloud(name + " singularities", field.tb.cycleSources[field.singLocalCycles])
    psSings.set_radius(radiusRatio*field.tb.avgAdjLength, relative=False)
    psSings.add_color_quantity("Indices", np.array([default_index_color(i) for i in field.singIndices]).reshape(-1, 3), enabled=True)
    return psSings


def set_cartesian_field(field, name="Field", unitToAvgLengthRatio=0.3):
    psGlyphs = set_raw_field(field.tb.sources, field.extField, name, unitToAvgLengthRatio*field.tb.avgAdjLength)
    set_singularities(field, name)
    return psGlyphs


def set_streamlines(state, name="Streamlines", width=0.05):
    numSegments = len(state.segStart)
    edges = np.column_stack([np.arange(numSegments), numSegments+np.arange(numSegments)])
    psStreamlines = ps.register_curve_network(name, np.vstack([state.segStart, state.segEnd]), edges)
    psStreamlines.set_radius(width, relative=False)
    return psStreamlines
