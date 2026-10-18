// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <directional/CartesianField.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/combing.h>
#include <directional/curl_matching.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::CartesianField;
using directional::fieldTypeEnum;

//combing() and curl_matching() walk the three edges of each face, so they only work on face-based fields.
inline void require_face_raw_field(const CartesianField& field)
{
    require(field.fieldType==fieldTypeEnum::RAW_FIELD, "field must be a RAW_FIELD");
    require(field.tb->discTangType()==directional::discTangTypeEnum::FACE_SPACES, "field must be face-based");
}

void bind_matching(nb::module_& m)
{
    m.def("combing", [](const CartesianField& rawField, const std::optional<Eigen::MatrixXi>& spaceIsCut){
        require_face_raw_field(rawField);
        require(rawField.matching.size()==rawField.tb->adjSpaces.rows(), "rawField has no matching (call principal_matching first)");
        require(!spaceIsCut || (spaceIsCut->rows()==rawField.tb->numSpaces && spaceIsCut->cols()==3),
                "spaceIsCut must be #F by 3");
        CartesianField combedField;
        directional::combing(rawField, combedField, spaceIsCut.value_or(Eigen::MatrixXi()));
        return combedField;
    }, "rawField"_a, "spaceIsCut"_a=nb::none(), nb::keep_alive<0, 1>());

    m.def("curl_matching", [](CartesianField& rawField){
        require_face_raw_field(rawField);
        Eigen::VectorXd curlNorm;
        directional::curl_matching(rawField, curlNorm);
        return curlNorm;
    }, "rawField"_a);
}
