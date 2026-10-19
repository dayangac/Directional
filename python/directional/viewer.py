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
DEFAULT_VECTOR_CONSTRAINT_COLOR = (1.0, 1.0, 0.2)
HIGHLIGHT_FACE_COLOR = (0.7, 0.2, 0.2)
#singularity colors for indices -3..3 (the index 0 color is the face color); lower is black, higher is white
INDEX_COLORS = {-3: (1.0, 1.0, 0.0), -2: (1.0, 0.5, 0.0), -1: (1.0, 0.0, 0.0), 0: DEFAULT_FACE_COLOR,
                1: (0.0, 1.0, 0.0), 2: (0.0, 1.0, 0.5), 3: (0.0, 1.0, 1.0)}
GLYPH_PRINCIPAL_COLORS = ((0.0, 0.5, 1.0), (1.0, 0.5, 0.0), (0.0, 1.0, 0.5), (1.0, 0.0, 0.5), (0.5, 0.0, 1.0),
                          (0.5, 1.0, 0.0), (1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0), (0.5, 1.0, 1.0),
                          (1.0, 0.5, 1.0), (1.0, 1.0, 0.5), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0))


def default_index_color(index):
    if index<=-4:
        return (0.0, 0.0, 0.0)
    if index>=4:
        return (1.0, 1.0, 1.0)
    return INDEX_COLORS[index]


#Colors by indices in each directional object. If the field is combed they will appear coherent across faces.
def indexed_glyph_colors(vectorNum, N, signSymmetry=True):
    return GLYPH_PRINCIPAL_COLORS[vectorNum%(N//2) if signSymmetry and N%2==0 else vectorNum]


def set_surface_mesh(mesh, name="Mesh"):
    psMesh = ps.register_surface_mesh(name, mesh.V, mesh.F, color=DEFAULT_FACE_COLOR)
    #Directional's edge order already matches polyscope's, which makes edge data (such as effort) displayable
    psMesh.set_edge_permutation(np.arange(len(mesh.EV)))
    return psMesh


#color is either one color for all vectors, or a list with a color per vector (polyscope cannot recolor vectors later,
#so this replaces DirectionalViewer::toggle_combed_colors())
def set_raw_field(sources, rawField, name="Field", unitLength=1.0, color=DEFAULT_GLYPH_COLOR):
    N = rawField.shape[1]//3
    colors = color if np.ndim(color)==2 else [color]*N
    psGlyphs = ps.register_point_cloud(name, sources, radius=10e-6, point_render_mode="quad")
    for i in range(N):
        psGlyphs.add_vector_quantity("Vector " + str(i), rawField[:, 3*i:3*i+3]*unitLength, enabled=True,
                                     vectortype="ambient", color=colors[i])
    return psGlyphs


#Shows the field's own singularities, unless singElements and singIndices are given.
def set_singularities(field, name="Field", radiusRatio=0.3, singElements=None, singIndices=None):
    if singElements is None:
        singElements, singIndices = field.singLocalCycles, field.singIndices
    psSings = ps.register_point_cloud(name + " singularities", field.tb.cycleSources[singElements])
    psSings.set_radius(radiusRatio*field.tb.avgAdjLength, relative=False)
    psSings.add_color_quantity("Indices", np.array([default_index_color(i) for i in singIndices]).reshape(-1, 3), enabled=True)
    return psSings


def set_cartesian_field(field, name="Field", unitToAvgLengthRatio=0.3, color=DEFAULT_GLYPH_COLOR):
    psGlyphs = set_raw_field(field.tb.sources, field.extField, name, unitToAvgLengthRatio*field.tb.avgAdjLength, color)
    set_singularities(field, name)
    return psGlyphs


def highlight_faces(faces, name="Face highlights", meshName="Mesh"):
    psMesh = ps.get_surface_mesh(meshName)
    faceColors = np.tile(psMesh.get_color(), (psMesh.n_faces(), 1))
    faceColors[faces] = HIGHLIGHT_FACE_COLOR
    psMesh.add_color_quantity(name, faceColors, defined_on="faces", enabled=True)


def highlight_edges(mesh, edges, name="Edge highlights", widthToEdgeLengthRatio=0.05):
    numEdges = len(edges)
    seamEdges = np.column_stack([np.arange(numEdges), numEdges+np.arange(numEdges)])
    psSeams = ps.register_curve_network(name, np.vstack([mesh.V[mesh.EV[edges, 0]], mesh.V[mesh.EV[edges, 1]]]), seamEdges, color=(0.0, 0.0, 0.0))
    psSeams.set_radius(widthToEdgeLengthRatio*mesh.avgEdgeLength, relative=False)
    return psSeams


def set_streamlines(state, name="Streamlines", width=0.05):
    numSegments = len(state.segStart)
    edges = np.column_stack([np.arange(numSegments), numSegments+np.arange(numSegments)])
    psStreamlines = ps.register_curve_network(name, np.vstack([state.segStart, state.segEnd]), edges)
    psStreamlines.set_radius(width, relative=False)
    return psStreamlines
