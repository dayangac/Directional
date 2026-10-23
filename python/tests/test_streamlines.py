# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

import gc
from pathlib import Path

import numpy as np
import pytest

import directional as dr

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"
N = 3
SEEDS = np.array([0, 300, 900])
NO_CONSTRAINTS = (np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))


def load_sphere():
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))


@pytest.fixture(scope="module")
def sphere():
    return load_sphere()


def raw_field(mesh, TangentBundle=dr.PCFaceTangentBundle):
    return dr.power_to_raw(dr.power_field(TangentBundle(mesh), *NO_CONSTRAINTS, N), N, normalize=True)


def trace(rawField, numSteps, dTime, seedLocations=SEEDS):
    data, state = dr.StreamlineData(), dr.StreamlineState()
    dr.streamlines_init(rawField, seedLocations, 3.0, data, state)
    for i in range(numSteps):
        dr.streamlines_next(data, state, dTime)
    return data, state


def barycentric_coordinates(mesh, faces, points):
    a, b, c = (mesh.V[mesh.F[faces, k]] for k in range(3))
    #solving points = a + u*(b-a) + v*(c-a) by normal equations (exact for points on the triangle plane)
    basis = np.stack([b-a, c-a], axis=2)
    uv = np.linalg.solve(np.einsum("nki,nkj->nij", basis, basis), np.einsum("nki,nk->ni", basis, points-a)[:, :, None])[:, :, 0]
    return np.column_stack([1-uv.sum(axis=1), uv])


def test_init_starts_N_streamlines_at_each_seed(sphere):
    data, state = trace(raw_field(sphere), 0, 0.0)
    np.testing.assert_array_equal(data.sampleFaces, SEEDS)
    np.testing.assert_array_equal(state.segOrigFace, np.tile(SEEDS, N))
    np.testing.assert_array_equal(state.segOrigVector, np.repeat(np.arange(N), len(SEEDS)))
    np.testing.assert_allclose(state.segStart, np.tile(sphere.barycenters[SEEDS], (N, 1)))


def test_first_step_follows_the_field(sphere):
    rawField = raw_field(sphere)
    dTime = 0.01*sphere.avgEdgeLength
    data, state = trace(rawField, 1, dTime)
    vectors = np.vstack([rawField.extField[SEEDS, 3*j:3*j+3] for j in range(N)])
    np.testing.assert_allclose(state.segEnd-state.segStart, dTime*vectors, atol=1e-12)
    assert state.currTime==pytest.approx(dTime)


def test_segments_are_signed_with_their_start_time(sphere):
    data, state = trace(raw_field(sphere), 20, 0.5*sphere.avgEdgeLength)
    times, lengths = state.segTimeSignatures, np.linalg.norm(state.segEnd-state.segStart, axis=1)
    np.testing.assert_array_equal(times[:N*len(SEEDS)], 0.0)
    assert np.all(times<=state.currTime)
    #the field has unit length, so the time a streamline enters a face is the time it entered the previous one plus the segment length
    for j in np.flatnonzero(times>0.0):
        previous = np.flatnonzero(np.all(state.segEnd==state.segStart[j], axis=1))
        assert len(previous)==1
        assert times[j]==pytest.approx(times[previous[0]]+lengths[previous[0]], abs=1e-12)


@pytest.mark.parametrize("TangentBundle", [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle])
def test_segments_stay_inside_their_faces(sphere, TangentBundle):
    data, state = trace(raw_field(sphere, TangentBundle), 20, 0.5*sphere.avgEdgeLength)
    assert len(state.segStart)>N*len(SEEDS)
    assert state.currTime==pytest.approx(10*sphere.avgEdgeLength)
    for points in (state.segStart, state.segEnd):
        np.testing.assert_allclose(np.sum((points-sphere.V[sphere.F[state.segOrigFace, 0]])*state.segNormal, axis=1), 0, atol=1e-12)
        assert barycentric_coordinates(sphere, state.segOrigFace, points).min()>-1e-6


def test_streamlines_stop_at_the_boundary():
    #lilium has a boundary, where streamlines must end instead of continuing in another face
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "lilium.obj"))
    rawField = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "lilium.rawfield"), dr.PCFaceTangentBundle(mesh))
    data, state = trace(rawField, 200, 0.5*mesh.avgEdgeLength, np.arange(0, len(mesh.F), 40))
    assert not np.all(state.segmentAlive)
    for points in (state.segStart, state.segEnd):
        assert barycentric_coordinates(mesh, state.segOrigFace, points).min()>-1e-6


def test_automatic_seeds_cover_the_mesh(sphere):
    data, state = trace(raw_field(sphere), 0, 0.0, seedLocations=np.empty(0, dtype=int))
    assert len(data.sampleFaces)>10
    #every seed starts N segments, except for the rare direction whose exit from the seed face is not found
    assert state.segmentAlive.size==N*len(data.sampleFaces)
    assert len(state.segStart)==np.count_nonzero(state.segmentAlive)


def test_data_outlives_python_references_to_field_and_mesh():
    data, state = trace(raw_field(load_sphere()), 0, 0.0)
    gc.collect()
    dr.streamlines_next(data, state, 0.1)
    assert len(state.segStart)>=N*len(SEEDS)


def test_streamlines_init_rejects_bad_input(sphere):
    rawField = raw_field(sphere)
    powerField = dr.power_field(dr.PCFaceTangentBundle(sphere), *NO_CONSTRAINTS, N)
    for args in [(powerField, SEEDS, 3.0), (rawField, np.array([len(sphere.F)]), 3.0), (rawField, np.empty(0, dtype=int), -1.0)]:
        with pytest.raises(ValueError):
            dr.streamlines_init(*args, dr.StreamlineData(), dr.StreamlineState())


def test_streamlines_next_rejects_uninitialized_or_mismatched_state(sphere):
    with pytest.raises(ValueError):
        dr.streamlines_next(dr.StreamlineData(), dr.StreamlineState(), 0.1)
    data, state = trace(raw_field(sphere), 0, 0.0)
    with pytest.raises(ValueError):
        dr.streamlines_next(data, dr.StreamlineState(), 0.1)
