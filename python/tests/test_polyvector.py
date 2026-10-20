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
N = 4


@pytest.fixture(scope="module")
def sphere():
    return dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))


@pytest.fixture(scope="module")
def ftb(sphere):
    return dr.PCFaceTangentBundle(sphere)


def complex_field(field):
    return field.intField[:, 0::2]+1j*field.intField[:, 1::2]


def pv_data(ftb, constSpaces, constVectors, wAlignment, N=N):
    pvData = dr.PolyVectorData(ftb, N)
    pvData.constSpaces = constSpaces
    pvData.constVectors = constVectors
    pvData.wAlignment = wAlignment
    return pvData


#edge directions: two on face 0 (a sign-symmetric 4-field has two free vectors per face), and one on two other faces
@pytest.fixture
def constraints(sphere):
    constSpaces = np.array([0, 0, 300, 900])
    edges = np.array([[0, 1], [0, 2], [0, 1], [1, 2]])
    F = sphere.F[constSpaces]
    constVectors = sphere.V[F[np.arange(4), edges[:, 1]]]-sphere.V[F[np.arange(4), edges[:, 0]]]
    return constSpaces, constVectors/np.linalg.norm(constVectors, axis=1)[:, None]


def test_polyvector_round_trip(ftb):
    rng = np.random.default_rng(0)
    for signSymmetry, roots in ((True, rng.normal(size=(ftb.numSpaces, 2))+1j*rng.normal(size=(ftb.numSpaces, 2))),
                                (False, rng.normal(size=(ftb.numSpaces, 3))+1j*rng.normal(size=(ftb.numSpaces, 3)))):
        if signSymmetry:
            roots = np.hstack([roots, -roots])
        rawField = dr.CartesianField(ftb, dr.fieldTypeEnum.RAW_FIELD, roots.shape[1])
        rawField.set_intrinsic_field(np.stack([roots.real, roots.imag], axis=2).reshape(ftb.numSpaces, -1))
        pvField = dr.raw_to_polyvector(rawField, signSymmetry)
        assert pvField.fieldType==dr.fieldTypeEnum.POLYVECTOR_FIELD
        backRoots = complex_field(dr.polyvector_to_raw(pvField, signSymmetry))
        #the same roots, in any order
        np.testing.assert_allclose(np.sort_complex(backRoots), np.sort_complex(roots), atol=1e-8)


def test_polyvector_hard_constraints(ftb, constraints):
    constSpaces, constVectors = constraints
    rawField = dr.polyvector_to_raw(dr.polyvector_field(pv_data(ftb, constSpaces, constVectors, np.full(len(constSpaces), -1.0))))
    for space, vector in zip(constSpaces, constVectors):
        distances = np.linalg.norm(rawField.extField[space].reshape(N, 3)-vector, axis=1)
        assert np.min(distances)<1e-8


def test_perfect_rosy_is_rotationally_symmetric(ftb, constraints):
    constSpaces, constVectors = constraints
    pvData = pv_data(ftb, constSpaces[2:], constVectors[2:], np.full(2, -1.0))
    pvData.wRoSy = -1.0
    vectors = complex_field(dr.polyvector_to_raw(dr.polyvector_field(pvData)))
    np.testing.assert_allclose(vectors[:, 1:], vectors[:, :-1]*np.exp(2j*np.pi/N), atol=1e-8)


def test_polyvector_singularities_satisfy_poincare_hopf(sphere, ftb, constraints):
    rawField = dr.polyvector_to_raw(dr.polyvector_field(pv_data(ftb, *constraints, np.ones(len(constraints[0])))))
    dr.principal_matching(rawField)
    assert np.sum(rawField.singIndices)==N*sphere.eulerChar


def iterated_field(ftb, constraints, iterationFunctions, numIterations, **kwargs):
    pvData = pv_data(ftb, *constraints, np.ones(len(constraints[0])))
    pvData.iterationMode = True
    pvField = dr.polyvector_field(pvData)
    dr.polyvector_iterate(pvData, pvField, iterationFunctions, numIterations, **kwargs)
    return pvData, pvField


def test_hard_rosy_iterations_give_unit_rosy_field(ftb, constraints):
    _, pvField = iterated_field(ftb, constraints, [dr.hard_rosy], 5)
    vectors = complex_field(dr.polyvector_to_raw(pvField))
    np.testing.assert_allclose(np.abs(vectors), 1.0, atol=1e-8)
    np.testing.assert_allclose(vectors[:, 1:], vectors[:, :-1]*np.exp(2j*np.pi/N), atol=1e-8)


def test_curl_projection_reduces_curl(ftb, constraints):
    pvData = pv_data(ftb, *constraints, np.ones(len(constraints[0])))
    pvData.wRoSy = 5.0
    curlOrig = dr.curl_matching(dr.polyvector_to_raw(dr.polyvector_field(pvData)))
    _, pvField = iterated_field(ftb, constraints, [dr.soft_rosy, dr.curl_projection], 10)
    curlFree = dr.curl_matching(dr.polyvector_to_raw(pvField))
    assert np.max(curlFree)<0.1*np.max(curlOrig)


def test_python_iteration_and_termination_functions(ftb, constraints):
    calls = []

    def logged_rosy(pvField, pvData):
        calls.append(pvData.currIteration)
        return dr.hard_rosy(pvField, pvData)

    pvData, _ = iterated_field(ftb, constraints, [logged_rosy], 10,
                               terminationFunction=lambda pvField, pvData: pvData.currIteration>=3)
    assert calls==[0, 1, 2]
    assert pvData.currIteration==3


def test_polyvector_data_defaults(ftb):
    pvData = dr.PolyVectorData(ftb, N)
    assert (pvData.signSymmetry, pvData.wSmooth, pvData.implicitFirst, pvData.currImplicitFactor)==(True, 1.0, True, 0.0)
    assert len(pvData.confidence)==0


def test_sign_symmetry_zeroes_the_odd_coefficients(ftb, constraints):
    for signSymmetry in (True, False):
        pvData = pv_data(ftb, *constraints, np.ones(len(constraints[0])))
        pvData.signSymmetry = signSymmetry
        oddCoefficients = complex_field(dr.polyvector_field(pvData))[:, 1::2]
        assert (np.max(np.abs(oddCoefficients))==0.0)==signSymmetry


def test_odd_N_has_no_sign_symmetry(ftb):
    pvData = pv_data(ftb, np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.ones(1), 3)
    dr.polyvector_field(pvData)
    assert not pvData.signSymmetry


def test_smoothness_weight_trades_off_alignment(sphere, ftb, constraints):
    constSpaces, constVectors = constraints
    errors = []
    for wSmooth in (0.1, 1.0, 10.0):
        pvData = pv_data(ftb, constSpaces, constVectors, np.ones(len(constSpaces)))
        pvData.wSmooth = wSmooth
        vectors = dr.polyvector_to_raw(dr.polyvector_field(pvData), normalizeRoots=True).extField[constSpaces].reshape(len(constSpaces), N, 3)
        #distance of each constraint from its closest vector
        errors.append(np.max(np.min(np.linalg.norm(vectors-constVectors[:, None, :], axis=2), axis=1)))
    assert errors[0]<errors[1]<errors[2]


def test_implicit_factor_follows_its_schedule(ftb, constraints):
    pvData, _ = iterated_field(ftb, constraints, [dr.hard_rosy], 0)
    initFactor = pvData.currImplicitFactor
    assert initFactor>0.0
    pvData.initImplicitFactor *= 2.0
    dr.polyvector_field(pvData)
    assert pvData.currImplicitFactor==pytest.approx(2.0*initFactor)
    pvField = dr.polyvector_field(pvData)
    dr.polyvector_iterate(pvData, pvField, [dr.hard_rosy], 3)
    assert pvData.currImplicitFactor==pytest.approx(2.0*initFactor*pvData.implicitScheduler**3)


def iterated_with(ftb, constraints, option, value):
    pvData = pv_data(ftb, *constraints, np.ones(len(constraints[0])))
    pvData.iterationMode = True
    setattr(pvData, option, value)
    pvField = dr.polyvector_field(pvData)
    dr.polyvector_iterate(pvData, pvField, [dr.hard_rosy], 5)
    return pvField


def test_implicit_step_last_smooths_the_projected_field(ftb, constraints):
    lengths = np.abs(complex_field(dr.polyvector_to_raw(iterated_with(ftb, constraints, "implicitFirst", False))))
    assert np.max(np.abs(lengths-1.0))>1e-4
    np.testing.assert_allclose(lengths, 1.0, atol=0.1)


def test_confidence_weighs_the_previous_iteration(ftb, constraints):
    _, pvField = iterated_field(ftb, constraints, [dr.hard_rosy], 5)
    #the default is a confidence of one, and the initial implicit factor cancels out any uniform confidence
    for value in (1.0, 0.5):
        uniformField = iterated_with(ftb, constraints, "confidence", np.full(ftb.numSpaces, value))
        np.testing.assert_allclose(uniformField.intField, pvField.intField, atol=1e-10)
    confidence = np.where(np.arange(ftb.numSpaces)<ftb.numSpaces//2, 1.0, 0.1)
    varyingField = iterated_with(ftb, constraints, "confidence", confidence)
    assert np.max(np.abs(varyingField.intField-pvField.intField))>1e-4
    np.testing.assert_allclose(np.abs(complex_field(dr.polyvector_to_raw(varyingField))), 1.0, atol=1e-8)


def test_polyvector_rejects_bad_input(sphere, ftb, constraints):
    constSpaces, constVectors = constraints
    with pytest.raises(ValueError):
        dr.PolyVectorData(ftb, 0)
    with pytest.raises(ValueError):
        dr.polyvector_field(pv_data(ftb, constSpaces, constVectors, np.ones(2)))
    with pytest.raises(ValueError):
        dr.polyvector_field(pv_data(ftb, constSpaces+ftb.numSpaces, constVectors, np.ones(len(constSpaces))))
    pvData = pv_data(ftb, *constraints, np.ones(len(constSpaces)))
    pvField = dr.polyvector_field(pvData)
    with pytest.raises(ValueError):  #iterationMode was not set
        dr.polyvector_iterate(pvData, pvField, [dr.hard_rosy])
    pvData.iterationMode = True
    pvData.confidence = np.ones(3)
    with pytest.raises(ValueError):
        dr.polyvector_field(pvData)
    with pytest.raises(ValueError):
        dr.polyvector_to_raw(dr.polyvector_to_raw(pvField))
    vtb = dr.IntrinsicVertexTangentBundle(sphere)
    pvData = pv_data(vtb, np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]))
    pvData.iterationMode = True
    pvField = dr.polyvector_field(pvData)
    with pytest.raises(ValueError):
        dr.polyvector_iterate(pvData, pvField, [dr.curl_projection])


def test_polyvector_field_keeps_bundle_alive(constraints):
    mesh = dr.readOBJ(str(TUTORIAL_DATA_PATH / "spherers.obj"))
    pvData = pv_data(dr.PCFaceTangentBundle(mesh), *constraints, np.ones(len(constraints[0])))
    pvField = dr.polyvector_field(pvData)
    del mesh, pvData
    gc.collect()
    assert dr.polyvector_to_raw(pvField).extField.shape==(1280, 3*N)
