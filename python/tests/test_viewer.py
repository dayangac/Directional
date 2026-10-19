# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np
import pytest

ps = pytest.importorskip("polyscope")
import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"
N = 4


@pytest.fixture(scope="module", autouse=True)
def polyscope_mock():
    ps.init("openGL_mock")
    yield
    ps.remove_all_structures()


@pytest.fixture(scope="module")
def rawField():
    mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "bumpy.off"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "bumpy.rawfield"), dr.PCFaceTangentBundle(mesh))
    dr.read_singularities(str(TUTORIAL_DATA_PATH / "bumpy.sings"), rawField)
    return rawField


def test_cartesian_field_shows_glyphs_and_singularities(rawField):
    psGlyphs = viewer.set_cartesian_field(rawField, "Bumpy")
    assert psGlyphs.n_points()==rawField.tb.numSpaces
    assert ps.get_point_cloud("Bumpy singularities").n_points()==len(rawField.singIndices)


def test_surface_mesh_accepts_edge_data(rawField):
    mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "bumpy.off"))
    psMesh = viewer.set_surface_mesh(mesh)
    psMesh.add_scalar_quantity("Edge index", np.arange(len(mesh.EV), dtype=float), defined_on="edges")


def test_index_colors_saturate():
    assert viewer.default_index_color(-7)==(0.0, 0.0, 0.0)
    assert viewer.default_index_color(9)==(1.0, 1.0, 1.0)
    assert viewer.default_index_color(1)==(0.0, 1.0, 0.0)


def test_streamlines_are_one_edge_per_segment():
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    rawField = dr.power_to_raw(dr.power_field(dr.PCFaceTangentBundle(mesh), np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0), 3), 3)
    data, state = dr.StreamlineData(), dr.StreamlineState()
    dr.streamlines_init(rawField, np.array([0, 10]), 3.0, data, state)
    dr.streamlines_next(data, state, 0.5*mesh.avgEdgeLength)
    psStreamlines = viewer.set_streamlines(state)
    assert psStreamlines.n_edges()==len(state.segStart)


def test_streamline_width_is_absolute():
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    rawField = dr.power_to_raw(dr.power_field(dr.PCFaceTangentBundle(mesh), np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0), 3), 3)
    data, state = dr.StreamlineData(), dr.StreamlineState()
    dr.streamlines_init(rawField, np.array([0, 10]), 3.0, data, state)
    dr.streamlines_next(data, state, 0.5*mesh.avgEdgeLength)
    assert viewer.set_streamlines(state, width=0.01).get_radius()==pytest.approx(0.01)


def test_combed_colors_follow_sign_symmetry():
    assert viewer.indexed_glyph_colors(2, 4)==viewer.indexed_glyph_colors(0, 4)
    assert viewer.indexed_glyph_colors(2, 4, signSymmetry=False)!=viewer.indexed_glyph_colors(0, 4, signSymmetry=False)
    assert viewer.indexed_glyph_colors(2, 3)!=viewer.indexed_glyph_colors(0, 3)


def test_raw_field_accepts_a_color_per_vector(rawField):
    viewer.set_raw_field(rawField.tb.sources, rawField.extField, "Combed", color=[viewer.indexed_glyph_colors(i, N) for i in range(N)])


def test_explicit_singularities(rawField):
    psSings = viewer.set_singularities(rawField, "Prescribed", singElements=np.array([3, 5, 8]), singIndices=np.array([1, -1, 2]))
    assert psSings.n_points()==3


def test_highlights():
    mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "bumpy.off"))
    viewer.set_surface_mesh(mesh)
    viewer.highlight_faces(np.array([0, 4]))
    psSeams = viewer.highlight_edges(mesh, np.array([1, 2, 7]))
    assert psSeams.n_edges()==3
    assert psSeams.get_radius()==pytest.approx(0.05*mesh.avgEdgeLength)
