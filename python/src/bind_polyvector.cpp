// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <directional/CartesianField.h>
#include <directional/PolyVectorData.h>
#include <directional/IntrinsicVertexTangentBundle.h>
#include <directional/polyvector_field.h>
#include <directional/polyvector_iteration_functions.h>
#include <directional/polyvector_to_raw.h>
#include <directional/raw_to_polyvector.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::CartesianField;
using directional::PolyVectorData;
using directional::TangentBundle;
using directional::fieldTypeEnum;

//PolyVectorData holds solvers and cannot be copied, so iteration and termination functions get the caller's objects by reference.
template <typename Result>
std::function<Result(const CartesianField&, const PolyVectorData&)> by_reference(const nb::callable& function)
{
    return [function](const CartesianField& pvField, const PolyVectorData& pvData){
        return nb::cast<Result>(function(nb::cast(pvField, nb::rv_policy::reference), nb::cast(pvData, nb::rv_policy::reference)));
    };
}

//conjugate() and conjugate_termination() read the principal curvatures through IntrinsicVertexTangentBundle.
inline void require_conjugate_field(const CartesianField& pvField, const PolyVectorData& pvData)
{
    require(pvField.N==4 && pvData.signSymmetry, "conjugate fields are sign-symmetric 4-fields");
    require(dynamic_cast<const directional::IntrinsicVertexTangentBundle*>(pvField.tb)!=nullptr, "conjugate fields need an IntrinsicVertexTangentBundle");
}

/***
 Binds the extended PolyVector algorithm. PolyVectorData keeps a pointer to its tangent bundle, and the fields it produces point to the same bundle (keep_alive).
 Iteration and termination functions can be the bound ones (hard_rosy, soft_rosy, curl_projection) or any Python callable with the same signature.
 ***/
void bind_polyvector(nb::module_& m)
{
    nb::class_<PolyVectorData>(m, "PolyVectorData")
    .def("__init__", [](PolyVectorData* pvData, const TangentBundle& tb, const int N){
        require(N>0, "N must be positive");
        new (pvData) PolyVectorData();
        pvData->tb = &tb;
        pvData->N = N;
    }, "tb"_a, "N"_a, nb::keep_alive<1, 2>())
    .def_prop_ro("tb", [](const PolyVectorData& pvData){ return pvData.tb; }, nb::rv_policy::reference)
    .def_ro("N", &PolyVectorData::N)
    .def_rw("constSpaces", &PolyVectorData::constSpaces)
    .def_rw("constVectors", &PolyVectorData::constVectors)
    .def_rw("verbose", &PolyVectorData::verbose)
    .def_rw("signSymmetry", &PolyVectorData::signSymmetry)
    .def_rw("wSmooth", &PolyVectorData::wSmooth)
    .def_rw("wRoSy", &PolyVectorData::wRoSy)
    .def_rw("wAlignment", &PolyVectorData::wAlignment)
    .def_rw("confidence", &PolyVectorData::confidence)
    .def_rw("initImplicitFactor", &PolyVectorData::initImplicitFactor)
    .def_rw("implicitScheduler", &PolyVectorData::implicitScheduler)
    .def_rw("iterationMode", &PolyVectorData::iterationMode)
    .def_rw("implicitFirst", &PolyVectorData::implicitFirst)
    .def_ro("currImplicitFactor", &PolyVectorData::currImplicitFactor)
    .def_ro("currIteration", &PolyVectorData::currIteration);

    m.def("polyvector_field", [](PolyVectorData& pvData){
        const int numConsts = pvData.constSpaces.size();
        require(pvData.constVectors.cols()==3, "constVectors must be #constSpaces by 3");
        require(pvData.constVectors.rows()==numConsts && pvData.wAlignment.size()==numConsts,
                "constSpaces, constVectors and wAlignment must have the same length");
        require(numConsts==0 || (pvData.constSpaces.minCoeff()>=0 && pvData.constSpaces.maxCoeff()<pvData.tb->numSpaces),
                "constSpaces contains out-of-range indices");
        require(pvData.confidence.size()==0 || pvData.confidence.size()==pvData.tb->numSpaces,
                "confidence must be empty or have a value per tangent space");
        CartesianField pvField;
        directional::polyvector_field(pvData, pvField);
        return pvField;
    }, "pvData"_a, nb::keep_alive<0, 1>());

    m.def("polyvector_iterate", [](PolyVectorData& pvData,
                                   CartesianField& pvField,
                                   const std::vector<nb::callable>& iterationFunctions,
                                   const int numIterations,
                                   const std::optional<nb::callable>& terminationFunction){
        require(pvData.iterationMode && pvData.implicitSolver.rows()>0,
                "pvData must be computed by polyvector_field with iterationMode set");
        require(pvField.fieldType==fieldTypeEnum::POLYVECTOR_FIELD && pvField.tb==pvData.tb && pvField.N==pvData.N,
                "pvField must be the POLYVECTOR_FIELD computed from pvData");
        std::vector<directional::PvIterationFunction> functions;
        for (const nb::callable& function : iterationFunctions)
            functions.push_back(by_reference<CartesianField>(function));
        directional::polyvector_iterate(pvData, pvField, functions, numIterations,
                                        (terminationFunction ? by_reference<bool>(*terminationFunction) : directional::default_termination));
    }, "pvData"_a, "pvField"_a, "iterationFunctions"_a, "numIterations"_a=1, "terminationFunction"_a=nb::none());

    m.def("hard_normalization", &directional::hard_normalization, "pvField"_a, "pvData"_a, nb::keep_alive<0, 1>());
    m.def("hard_rosy", &directional::hard_rosy, "pvField"_a, "pvData"_a, nb::keep_alive<0, 1>());
    m.def("soft_rosy", &directional::soft_rosy, "pvField"_a, "pvData"_a, nb::keep_alive<0, 1>());
    m.def("curl_projection", [](const CartesianField& pvField, const PolyVectorData& pvData){
        require(pvData.tb->discTangType()==directional::discTangTypeEnum::FACE_SPACES, "curl_projection only works on face-based fields");
        return directional::curl_projection(pvField, pvData);
    }, "pvField"_a, "pvData"_a, nb::keep_alive<0, 1>());
    m.def("conjugate", [](const CartesianField& pvField, const PolyVectorData& pvData){
        require_conjugate_field(pvField, pvData);
        return directional::conjugate(pvField, pvData);
    }, "pvField"_a, "pvData"_a, nb::keep_alive<0, 1>());
    m.def("default_termination", &directional::default_termination, "pvField"_a, "pvData"_a);
    m.def("conjugate_termination", [](const CartesianField& pvField, const PolyVectorData& pvData){
        require_conjugate_field(pvField, pvData);
        return directional::conjugate_termination(pvField, pvData);
    }, "pvField"_a, "pvData"_a);

    m.def("polyvector_to_raw", [](const CartesianField& pvField, const bool signSymmetry, const bool normalizeRoots, const double rootTolerance){
        require(pvField.fieldType==fieldTypeEnum::POLYVECTOR_FIELD, "pvField must be a POLYVECTOR_FIELD");
        CartesianField rawField;
        directional::polyvector_to_raw(pvField, rawField, signSymmetry, normalizeRoots, rootTolerance);
        return rawField;
    }, "pvField"_a, "signSymmetry"_a=true, "normalizeRoots"_a=false, "rootTolerance"_a=1e-8, nb::keep_alive<0, 1>());

    m.def("raw_to_polyvector", [](const CartesianField& rawField, const bool signSymmetry){
        require(rawField.fieldType==fieldTypeEnum::RAW_FIELD, "rawField must be a RAW_FIELD");
        CartesianField pvField;
        directional::raw_to_polyvector(rawField, pvField, signSymmetry);
        return pvField;
    }, "rawField"_a, "signSymmetry"_a=true, nb::keep_alive<0, 1>());
}
