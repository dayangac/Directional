// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <directional/CartesianField.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/setup_integration.h>
#include <directional/integrate.h>
#include <directional/branched_isolines.h>
#include <directional/branched_gradient.h>
#include <directional/cut_mesh_with_singularities.h>
#include <directional/isolines.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::CartesianField;
using directional::IntegrationData;
using directional::TriMesh;

//The integration only works on face-based fields of the same degree as intData.
inline void require_face_field(const CartesianField& field, const IntegrationData& intData)
{
    require(field.tb->discTangType()==directional::discTangTypeEnum::FACE_SPACES, "field must be face-based");
    require(field.N==intData.N, "field.N must equal intData.N");
}

/***
 Binds seamless integration. setup_integration() returns (meshCut, combedField), and integrate() returns
 (NFunction, NCornerFunctions) and raises if the solve fails.
 ***/
void bind_integration(nb::module_& m)
{
    nb::class_<IntegrationData>(m, "IntegrationData")
    .def("__init__", [](IntegrationData* intData, const int N){
        require(N>0, "N must be positive");
        new (intData) IntegrationData(N);
    }, "N"_a)
    .def_ro("N", &IntegrationData::N)
    .def_ro("n", &IntegrationData::n)
    .def_prop_ro("linRed", view(&IntegrationData::linRed), nb::rv_policy::reference_internal)
    .def_prop_ro("periodMat", view(&IntegrationData::periodMat), nb::rv_policy::reference_internal)
    .def_prop_ro("face2cut", view(&IntegrationData::face2cut), nb::rv_policy::reference_internal)
    .def_prop_ro("constrainedVertices", view(&IntegrationData::constrainedVertices), nb::rv_policy::reference_internal)
    .def_prop_ro("integerVars", view(&IntegrationData::integerVars), nb::rv_policy::reference_internal)
    .def_prop_ro("singularIndices", view(&IntegrationData::singularIndices), nb::rv_policy::reference_internal)
    .def_prop_ro("nVertexFunction", view(&IntegrationData::nVertexFunction), nb::rv_policy::reference_internal)
    //the translation that setup_integration() fixes (the first singularity, or vertex 0), which can be changed before integrate()
    .def_prop_rw("fixedIndices", view(&IntegrationData::fixedIndices), [](IntegrationData& intData, const Eigen::VectorXi& fixedIndices){
        require(fixedIndices.size()==intData.fixedValues.size(), "fixedIndices must have the length of fixedValues");
        intData.fixedIndices = fixedIndices;
    }, nb::rv_policy::reference_internal)
    .def_prop_rw("fixedValues", view(&IntegrationData::fixedValues), [](IntegrationData& intData, const Eigen::VectorXd& fixedValues){
        require(fixedValues.size()==intData.fixedIndices.size(), "fixedValues must have the length of fixedIndices");
        intData.fixedValues = fixedValues;
    }, nb::rv_policy::reference_internal)
    .def_prop_ro("vertexTrans2CutMat", [](const IntegrationData& intData){ return intData.vertexTrans2CutMat; })
    .def_prop_ro("constraintMat", [](const IntegrationData& intData){ return intData.constraintMat; })
    .def_prop_ro("linRedMat", [](const IntegrationData& intData){ return intData.linRedMat; })
    .def_prop_ro("intSpanMat", [](const IntegrationData& intData){ return intData.intSpanMat; })
    .def_prop_ro("singIntSpanMat", [](const IntegrationData& intData){ return intData.singIntSpanMat; })
    .def_rw("lengthRatio", &IntegrationData::lengthRatio)
    .def_rw("integralSeamless", &IntegrationData::integralSeamless)
    .def_rw("roundSeams", &IntegrationData::roundSeams)
    .def_rw("verbose", &IntegrationData::verbose)
    .def("set_linear_reduction", [](IntegrationData& intData, const Eigen::MatrixXi& linRed, const Eigen::MatrixXi& periodMat){
        require(linRed.size()>0 && periodMat.rows()==linRed.cols() && periodMat.cols()==linRed.cols(),
                "periodMat must be n by n for an N by n linRed");
        intData.set_linear_reduction(linRed, periodMat);
    }, "linRed"_a, "periodMat"_a)
    .def("set_sign_symmetry", [](IntegrationData& intData, const int N){
        require(N==intData.N && N%2==0, "N must be intData.N, and even");
        intData.set_sign_symmetry(N);
    }, "N"_a)
    .def("set_triangular_symmetry", [](IntegrationData& intData, const int N){
        require(N==intData.N && N%3==0, "N must be intData.N, and divisible by 3");
        intData.set_triangular_symmetry(N);
    }, "N"_a)
    .def("set_default_period_matrix", [](IntegrationData& intData, const int n){
        require(n==intData.n, "n must be intData.n");
        intData.set_default_period_matrix(n);
    }, "n"_a);

    //The combed field points to the bundle of the input field, so it keeps that field alive.
    m.def("setup_integration", [](nb::handle_t<CartesianField> fieldObject, IntegrationData& intData){
        const CartesianField& field = nb::cast<const CartesianField&>(fieldObject);
        require_face_field(field, intData);
        require(field.fieldType==directional::fieldTypeEnum::RAW_FIELD, "field must be a RAW_FIELD");
        require(field.matching.size()==field.tb->adjSpaces.rows(), "field has no matching (call principal_matching first)");
        TriMesh meshCut;
        CartesianField combedField;
        directional::setup_integration(field, intData, meshCut, combedField);
        nb::object combedObject = nb::cast(std::move(combedField));
        nb::keep_alive_obj(combedObject, fieldObject);
        return nb::typed<nb::tuple, TriMesh, CartesianField>(nb::make_tuple(std::move(meshCut), combedObject));
    }, "field"_a, "intData"_a);

    m.def("integrate", [](const CartesianField& field, IntegrationData& intData, const TriMesh& meshCut){
        require_face_field(field, intData);
        require(meshCut.F.rows()==field.tb->numSpaces, "meshCut must be face-corresponding with the mesh of the field");
        require(intData.vertexTrans2CutMat.rows()==intData.N*meshCut.V.rows() &&
                intData.linRedMat.rows()*intData.n==intData.linRedMat.cols()*intData.N,
                "intData must be set up by setup_integration() with this meshCut and linear reduction");
        Eigen::MatrixXd NFunction, NCornerFunctions;
        if (!directional::integrate(field, intData, meshCut, NFunction, NCornerFunctions))
            throw std::runtime_error("integration failed");
        return std::make_pair(NFunction, NCornerFunctions);
    }, "field"_a, "intData"_a, "meshCut"_a);

    m.def("cut_mesh_with_singularities", [](const TriMesh& mesh, const Eigen::VectorXi& singularities){
        require(singularities.size()==0 || (singularities.minCoeff()>=0 && singularities.maxCoeff()<mesh.V.rows()), "singularities contains out-of-range vertices");
        Eigen::MatrixXi face2cut;
        directional::cut_mesh_with_singularities(mesh, singularities, face2cut);
        return face2cut;
    }, "mesh"_a, "singularities"_a);

    //The gradient of N functions on the vertices (packets of N per vertex), as packets of N 3D vectors per face.
    m.def("branched_gradient", [](const TriMesh& mesh, const int N){
        require(N>0, "N must be positive");
        Eigen::SparseMatrix<double> G;
        directional::branched_gradient(mesh, N, G);
        return G;
    }, "mesh"_a, "N"_a);

    //Returns (isoV, isoE, isoOrigE, isoN) at the integer values of z. The C++ number of isolines is left out, since
    //the integer values are always used.
    m.def("isolines", [](const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& z){
        require(V.cols()==3 && F.cols()==3, "V and F must be #V by 3 and #F by 3");
        require(z.size()==V.rows(), "z must have a value per vertex");
        require(F.size()==0 || (F.minCoeff()>=0 && F.maxCoeff()<V.rows()), "F contains out-of-range vertex indices");
        Eigen::MatrixXd isoV, isoN;
        Eigen::MatrixXi isoE, isoOrigE;
        directional::isolines(V, F, z, 0, isoV, isoE, isoOrigE, isoN);
        return std::make_tuple(isoV, isoE, isoOrigE, isoN);
    }, "V"_a, "F"_a, "z"_a);

    //Returns (isoV, isoE, isoOrigE, isoN, funcNum).
    m.def("branched_isolines", [](const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& NFunction){
        require(V.cols()==3 && F.cols()==3, "V and F must be #V by 3 and #F by 3");
        require(NFunction.rows()==V.rows() && NFunction.cols()>0, "NFunction must have a row per vertex");
        Eigen::MatrixXd isoV, isoN;
        Eigen::MatrixXi isoE, isoOrigE;
        Eigen::VectorXi funcNum;
        directional::branched_isolines(V, F, NFunction, isoV, isoE, isoOrigE, isoN, funcNum);
        return std::make_tuple(isoV, isoE, isoOrigE, isoN, funcNum);
    }, "V"_a, "F"_a, "NFunction"_a);
}
