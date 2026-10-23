// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <numbers>
#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <directional/CartesianField.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/combing.h>
#include <directional/curl_matching.h>
#include <directional/effort_to_indices.h>
#include <directional/index_prescription.h>
#include <directional/project_curl.h>
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

    //Sets the singularities of a matched field from its effort.
    m.def("effort_to_indices", [](CartesianField& field){
        require(field.effort.size()==field.tb->adjSpaces.rows(), "field has no effort (call a matching function first)");
        directional::effort_to_indices(field);
    }, "field"_a);

    m.def("effort_to_indices", [](const Eigen::SparseMatrix<double>& basisCycles, const Eigen::VectorXd& effort, const Eigen::VectorXd& cycleCurvature, const int N){
        require(basisCycles.cols()==effort.size() && basisCycles.rows()==cycleCurvature.size(),
                "basisCycles must be #cycles by #effort, and cycleCurvature must have a value per cycle");
        const Eigen::VectorXd dIndices = (basisCycles*effort+N*cycleCurvature)/(2.0*std::numbers::pi);
        require(dIndices.size()==0 || (dIndices.array()-dIndices.array().round()).abs().maxCoeff()<1e-6, "indices are not naturally integer");
        Eigen::VectorXi indices;
        directional::effort_to_indices(basisCycles, effort, cycleCurvature, N, indices);
        return indices;
    }, "basisCycles"_a, "effort"_a, "cycleCurvature"_a, "N"_a);

    //The hard constraints are not implemented in the library yet, so constFaces and constVectors have no effect.
    m.def("project_curl", [](const CartesianField& origField, const Eigen::VectorXi& constFaces, const Eigen::MatrixXd& constVectors,
                             const std::optional<Eigen::SparseMatrix<double>>& objMatrix, const std::optional<Eigen::VectorXd>& objRhs,
                             const std::optional<Eigen::SparseMatrix<double>>& reducMatrix){
        require_face_raw_field(origField);
        require(origField.matching.size()==origField.tb->adjSpaces.rows(), "origField has no matching (call principal_matching or curl_matching first)");
        require(constVectors.rows()==constFaces.size() && (constFaces.size()==0 || constVectors.cols()==3), "constVectors must be #constFaces by 3");
        require(constFaces.size()==0 || (constFaces.minCoeff()>=0 && constFaces.maxCoeff()<origField.tb->numSpaces), "constFaces contains out-of-range faces");
        const int numDofs = 2*origField.N*origField.tb->numSpaces;
        require(objMatrix.has_value()==objRhs.has_value(), "objMatrix and objRhs must be given together");
        require(!objMatrix || (objMatrix->cols()==numDofs && objRhs->size()==objMatrix->rows()),
                "objMatrix must have a column per intrinsic degree of freedom, and objRhs a value per row");
        require(!reducMatrix || reducMatrix->rows()==numDofs, "reducMatrix must have a row per intrinsic degree of freedom");
        CartesianField curlFreeField;
        directional::project_curl(origField, constFaces, constVectors, curlFreeField, objMatrix.value_or(Eigen::SparseMatrix<double>()),
                                  objRhs.value_or(Eigen::VectorXd()), reducMatrix.value_or(Eigen::SparseMatrix<double>()));
        return curlFreeField;
    }, "origField"_a, "constFaces"_a=Eigen::VectorXi(), "constVectors"_a=Eigen::MatrixXd(0, 3), "objMatrix"_a=nb::none(),
       "objRhs"_a=nb::none(), "reducMatrix"_a=nb::none(), nb::keep_alive<0, 1>());

    //Returns (rotationAngles, linfError).
    m.def("index_prescription", [](const Eigen::VectorXi& cycleIndices, const int N, const double globalRotation, CartesianField& field){
        require(N>0, "N must be positive");
        require(cycleIndices.size()==field.tb->cycles.rows(), "cycleIndices must have an index per cycle of the tangent bundle");
        Eigen::VectorXd rotationAngles;
        double linfError;
        directional::index_prescription(cycleIndices, N, globalRotation, field, rotationAngles, linfError);
        return std::make_pair(rotationAngles, linfError);
    }, "cycleIndices"_a, "N"_a, "globalRotation"_a, "field"_a);
}
