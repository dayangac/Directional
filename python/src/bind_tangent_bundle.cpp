// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <directional/TangentBundle.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/IntrinsicVertexTangentBundle.h>
#include <directional/ExtrinsicVertexTangentBundle.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::TangentBundle;

//Checks the tangent spaces and the number of columns given to project_to_intrinsic() and project_to_extrinsic().
inline void require_spaces(const TangentBundle& tb, const Eigen::VectorXi& tangentSpaces, const int numDirectionals, const int dim)
{
    require(tangentSpaces.size()==0 || (tangentSpaces.minCoeff()>=0 && tangentSpaces.maxCoeff()<tb.numSpaces), "tangentSpaces contains out-of-range indices");
    require(numDirectionals%dim==0, dim==3 ? "extDirectionals must have 3 columns per vector" : "intDirectionals must have 2 columns per vector");
}

//Bundles keep a pointer to their mesh, so the mesh must outlive them (keep_alive).
template <typename Bundle>
void bind_bundle(nb::module_& m, const char* name)
{
    nb::class_<Bundle, TangentBundle>(m, name)
    .def("__init__", [](Bundle* tb, const directional::TriMesh& mesh){
        new (tb) Bundle();
        tb->init(mesh);
    }, "mesh"_a, nb::keep_alive<1, 2>())
    .def_prop_ro("mesh", [](const Bundle& tb){ return tb.mesh; }, nb::rv_policy::reference)
    .def("hasCochainSequence", &Bundle::hasCochainSequence)
    //Returns (interpSources, interpNormals, interpField) at points given by faces and barycentric coordinates.
    //intDirectionals is the whole intrinsic field (#spaces by 2N).
    .def("interpolate", [](const Bundle& tb, const Eigen::VectorXi& faces, const Eigen::MatrixXd& baryCoords, const Eigen::MatrixXd& intDirectionals){
        require(baryCoords.rows()==faces.size() && baryCoords.cols()==3, "baryCoords must be #faces by 3");
        require(faces.size()==0 || (faces.minCoeff()>=0 && faces.maxCoeff()<tb.mesh->F.rows()), "faces contains out-of-range indices");
        require(intDirectionals.rows()==tb.numSpaces && intDirectionals.cols()%2==0 && intDirectionals.cols()>0,
                "intDirectionals must be the whole intrinsic field, #spaces by 2N");
        Eigen::MatrixXd interpSources, interpNormals, interpField;
        tb.interpolate(faces, baryCoords, intDirectionals, interpSources, interpNormals, interpField);
        return std::make_tuple(interpSources, interpNormals, interpField);
    }, "faces"_a, "baryCoords"_a, "intDirectionals"_a);
}

/***
 Binds the TangentBundle interface, and the face-based and vertex-based bundles that implement it.
 The bundle arrays are read-only NumPy views, and the sparse matrices (cycles, masses) are scipy.sparse copies.
 ***/
void bind_tangent_bundle(nb::module_& m)
{
    nb::enum_<directional::discTangTypeEnum>(m, "discTangTypeEnum")
    .value("BASE_CLASS", directional::discTangTypeEnum::BASE_CLASS)
    .value("FACE_SPACES", directional::discTangTypeEnum::FACE_SPACES)
    .value("VERTEX_SPACES", directional::discTangTypeEnum::VERTEX_SPACES);

    nb::class_<TangentBundle>(m, "TangentBundle")
    .def("discTangType", &TangentBundle::discTangType)
    .def("hasEmbedding", &TangentBundle::hasEmbedding)
    .def_ro("intDimension", &TangentBundle::intDimension)
    .def_ro("numSpaces", &TangentBundle::numSpaces)
    .def_ro("avgAdjLength", &TangentBundle::avgAdjLength)
    .def_prop_ro("sources", view(&TangentBundle::sources), nb::rv_policy::reference_internal)
    .def_prop_ro("normals", view(&TangentBundle::normals), nb::rv_policy::reference_internal)
    .def_prop_ro("adjSpaces", view(&TangentBundle::adjSpaces), nb::rv_policy::reference_internal)
    .def_prop_ro("cycleSources", view(&TangentBundle::cycleSources), nb::rv_policy::reference_internal)
    .def_prop_ro("cycleCurvatures", view(&TangentBundle::cycleCurvatures), nb::rv_policy::reference_internal)
    .def_prop_ro("local2Cycle", view(&TangentBundle::local2Cycle), nb::rv_policy::reference_internal)
    .def_prop_ro("cycles", [](const TangentBundle& tb){ return tb.cycles; })  //a scipy.sparse copy
    .def_prop_ro("oneRing", view(&TangentBundle::oneRing), nb::rv_policy::reference_internal)
    .def_prop_ro("innerAdjacencies", view(&TangentBundle::innerAdjacencies), nb::rv_policy::reference_internal)
    .def_prop_ro("cycleNormals", view(&TangentBundle::cycleNormals), nb::rv_policy::reference_internal)
    .def_prop_ro("connection", view(&TangentBundle::connection), nb::rv_policy::reference_internal)
    .def_prop_ro("connectionMass", [](const TangentBundle& tb){ return tb.connectionMass; })
    .def_prop_ro("tangentSpaceMass", [](const TangentBundle& tb){ return tb.tangentSpaceMass; })
    .def_prop_ro("invTangentSpaceMass", [](const TangentBundle& tb){ return tb.invTangentSpaceMass; })
    .def("project_to_intrinsic", [](const TangentBundle& tb, const Eigen::VectorXi& tangentSpaces, const Eigen::MatrixXd& extDirectionals){
        require_spaces(tb, tangentSpaces, extDirectionals.cols(), 3);
        require(tangentSpaces.size()==extDirectionals.rows(), "extDirectionals must have a row per tangent space");
        return tb.project_to_intrinsic(tangentSpaces, extDirectionals);
    }, "tangentSpaces"_a, "extDirectionals"_a)
    //empty tangentSpaces means all of them
    .def("project_to_extrinsic", [](const TangentBundle& tb, const Eigen::VectorXi& tangentSpaces, const Eigen::MatrixXd& intDirectionals){
        require_spaces(tb, tangentSpaces, intDirectionals.cols(), 2);
        require(intDirectionals.rows()==(tangentSpaces.size()==0 ? tb.numSpaces : tangentSpaces.size()),
                "intDirectionals must have a row per tangent space (or per space of the bundle when tangentSpaces is empty)");
        return tb.project_to_extrinsic(tangentSpaces, intDirectionals);
    }, "tangentSpaces"_a, "intDirectionals"_a);

    bind_bundle<directional::PCFaceTangentBundle>(m, "PCFaceTangentBundle");
    bind_bundle<directional::IntrinsicVertexTangentBundle>(m, "IntrinsicVertexTangentBundle");
    bind_bundle<directional::ExtrinsicVertexTangentBundle>(m, "ExtrinsicVertexTangentBundle");
}
