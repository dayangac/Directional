import gc
from pathlib import Path

import numpy as np
import pytest

import directional as dr

DATA = Path(__file__).parents[2] / "tutorial" / "data"
N = 4
NO_CONSTRAINTS = (np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))


def load(name):
    return dr.readOBJ(str(DATA / name))


@pytest.fixture(scope="module")
def sphere():
    return load("spherers.obj")


def test_mesh_topology(sphere):
    assert sphere.V.shape == (642, 3)
    assert sphere.F.shape == (1280, 3)
    assert sphere.eulerChar == 2
    assert load("torus.obj").eulerChar == 0


def test_mesh_from_arrays_matches_file(sphere):
    mesh = dr.TriMesh(sphere.V, sphere.F)
    np.testing.assert_array_equal(mesh.EV, sphere.EV)
    np.testing.assert_allclose(mesh.faceNormals, sphere.faceNormals)


def test_array_properties_are_read_only(sphere):
    raw = dr.power_to_raw(dr.power_field(dr.PCFaceTangentBundle(sphere), *NO_CONSTRAINTS, N), N)
    for array in (sphere.V, sphere.F, dr.PCFaceTangentBundle(sphere).sources, raw.extField):
        with pytest.raises(ValueError):
            array[0] = 0


@pytest.mark.parametrize("bad_faces", [
    lambda F, n: F[:, :2],
    lambda F, n: F + n,
    lambda F, n: F - 1,
])
def test_mesh_rejects_bad_faces(sphere, bad_faces):
    with pytest.raises(ValueError):
        dr.TriMesh(sphere.V, bad_faces(sphere.F, len(sphere.V)))


def test_readOBJ_missing_file():
    with pytest.raises(RuntimeError):
        dr.readOBJ(str(DATA / "missing.obj"))


@pytest.mark.parametrize("Bundle", [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle])
def test_bundle_spaces(sphere, Bundle):
    tb = Bundle(sphere)
    expected = len(sphere.F) if Bundle is dr.PCFaceTangentBundle else len(sphere.V)
    assert tb.numSpaces == expected
    assert tb.sources.shape == tb.normals.shape == (expected, 3)


def test_hard_constraints_are_met(sphere):
    faces = np.array([0, 100, 500])
    vectors = sphere.V[sphere.F[faces, 1]] - sphere.V[sphere.F[faces, 0]]
    tb = dr.PCFaceTangentBundle(sphere)
    power = dr.power_field(tb, faces, vectors, -np.ones(len(faces)), N)
    raw = dr.power_to_raw(power, N)

    candidates = raw.extField[faces].reshape(len(faces), N, 3)
    dist = np.linalg.norm(candidates - vectors[:, None, :], axis=2).min(axis=1)
    np.testing.assert_allclose(dist, 0, atol=1e-8)


@pytest.mark.parametrize("Bundle", [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle])
def test_raw_field_is_rotationally_symmetric(sphere, Bundle):
    power = dr.power_field(Bundle(sphere), *NO_CONSTRAINTS, N)
    raw = dr.power_to_raw(power, N)
    assert raw.fieldType == dr.fieldTypeEnum.RAW_FIELD
    assert raw.intField.shape == (power.intField.shape[0], 2 * N)

    z = raw.intField[:, 0::2] + 1j * raw.intField[:, 1::2]
    np.testing.assert_allclose(z[:, 1:], z[:, :-1] * np.exp(2j * np.pi / N), atol=1e-10)


@pytest.mark.parametrize("name", ["spherers.obj", "torus.obj"])
@pytest.mark.parametrize("Bundle", [dr.PCFaceTangentBundle, dr.IntrinsicVertexTangentBundle])
def test_singularities_satisfy_poincare_hopf(name, Bundle):
    mesh = load(name)
    raw = dr.power_to_raw(dr.power_field(Bundle(mesh), *NO_CONSTRAINTS, N), N)
    dr.principal_matching(raw)
    assert len(raw.matching) == len(mesh.EV)
    assert raw.singIndices.sum() == N * mesh.eulerChar


def test_normalized_field_has_unit_vectors(sphere):
    power = dr.power_field(dr.PCFaceTangentBundle(sphere), *NO_CONSTRAINTS, N, normalizeField=True)
    np.testing.assert_allclose(np.linalg.norm(power.intField, axis=1), 1)


def test_power_field_types(sphere):
    powerField = dr.power_field(dr.PCFaceTangentBundle(sphere), np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), 4)
    assert powerField.fieldType==dr.fieldTypeEnum.POWER_FIELD
    assert dr.power_to_raw(powerField, 4).fieldType==dr.fieldTypeEnum.RAW_FIELD


@pytest.mark.parametrize("args", [
    (np.array([0, 1]), np.ones((1, 3)), -np.ones(2), N),
    (np.array([0]), np.ones((1, 2)), -np.ones(1), N),
    (np.array([10**6]), np.ones((1, 3)), -np.ones(1), N),
    (*NO_CONSTRAINTS, 0),
])
def test_power_field_rejects_bad_input(sphere, args):
    with pytest.raises(ValueError):
        dr.power_field(dr.PCFaceTangentBundle(sphere), *args)


def test_set_field_projects_between_representations(sphere):
    tb = dr.PCFaceTangentBundle(sphere)
    raw = dr.power_to_raw(dr.power_field(tb, *NO_CONSTRAINTS, N), N)

    from_int = dr.CartesianField(tb, dr.fieldTypeEnum.RAW_FIELD, N)
    from_int.set_intrinsic_field(raw.intField)
    np.testing.assert_allclose(from_int.extField, raw.extField, atol=1e-12)

    from_ext = dr.CartesianField(tb, dr.fieldTypeEnum.RAW_FIELD, N)
    from_ext.set_extrinsic_field(raw.extField)
    np.testing.assert_allclose(from_ext.intField, raw.intField, atol=1e-12)


def test_set_field_rejects_wrong_shape(sphere):
    field = dr.CartesianField(dr.PCFaceTangentBundle(sphere), dr.fieldTypeEnum.RAW_FIELD, N)
    with pytest.raises(ValueError):
        field.set_intrinsic_field(np.zeros((len(sphere.F), 2)))
    with pytest.raises(ValueError):
        field.set_extrinsic_field(np.zeros((len(sphere.F) - 1, 3 * N)))


def test_principal_matching_rejects_power_field(sphere):
    power = dr.power_field(dr.PCFaceTangentBundle(sphere), *NO_CONSTRAINTS, N)
    with pytest.raises(ValueError):
        dr.principal_matching(power)


def test_field_outlives_python_references_to_mesh_and_bundle():
    def make():
        tb = dr.PCFaceTangentBundle(load("spherers.obj"))
        return dr.power_to_raw(dr.power_field(tb, *NO_CONSTRAINTS, N), N)

    raw = make()
    gc.collect()
    dr.principal_matching(raw)
    assert raw.singIndices.sum() == 2 * N


def test_raw_field_file_roundtrip(sphere, tmp_path):
    tb = dr.PCFaceTangentBundle(sphere)
    raw = dr.power_to_raw(dr.power_field(tb, *NO_CONSTRAINTS, N), N)
    path = str(tmp_path / "field.rawfield")
    dr.write_raw_field(path, raw, high_precision=True)

    loaded = dr.read_raw_field(path, tb)
    assert loaded.N == N
    np.testing.assert_allclose(loaded.extField, raw.extField, atol=1e-12)


def test_read_raw_field_missing_file(sphere):
    with pytest.raises(RuntimeError):
        dr.read_raw_field(str(DATA / "missing.rawfield"), dr.PCFaceTangentBundle(sphere))
