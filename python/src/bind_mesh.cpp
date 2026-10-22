// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <directional/TriMesh.h>
#include <directional/readOBJ.h>
#include <directional/readOFF.h>
#include <directional/readDMAT.h>
#include <directional/writeOBJ.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::TriMesh;

//Wraps readOBJ/readOFF to return the mesh, and raise on failure instead of returning false.
template <bool (*read)(std::string, TriMesh&)>
TriMesh read_mesh(const std::string& fileName)
{
    TriMesh mesh;
    if (!read(fileName, mesh))
        throw std::runtime_error("could not read " + fileName);
    return mesh;
}

void bind_mesh(nb::module_& m)
{
    nb::class_<TriMesh>(m, "TriMesh")
    .def("__init__", [](TriMesh* mesh, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F){
        require(V.cols()==3 && F.cols()==3 && F.rows()>0, "V must be #V by 3 and F a non-empty #F by 3");
        require(F.minCoeff()>=0 && F.maxCoeff()<V.rows(), "F contains out-of-range vertex indices");
        new (mesh) TriMesh();
        mesh->set_mesh(V, F);
    }, "V"_a, "F"_a)
    .def_prop_ro("V", view(&TriMesh::V), nb::rv_policy::reference_internal)
    .def_prop_ro("F", view(&TriMesh::F), nb::rv_policy::reference_internal)
    .def_prop_ro("EV", view(&TriMesh::EV), nb::rv_policy::reference_internal)
    .def_prop_ro("EF", view(&TriMesh::EF), nb::rv_policy::reference_internal)
    .def_prop_ro("FE", view(&TriMesh::FE), nb::rv_policy::reference_internal)
    .def_prop_ro("FEs", view(&TriMesh::FEs), nb::rv_policy::reference_internal)
    .def_prop_ro("faceNormals", view(&TriMesh::faceNormals), nb::rv_policy::reference_internal)
    .def_prop_ro("vertexNormals", view(&TriMesh::vertexNormals), nb::rv_policy::reference_internal)
    .def_prop_ro("faceAreas", view(&TriMesh::faceAreas), nb::rv_policy::reference_internal)
    .def_prop_ro("barycenters", view(&TriMesh::barycenters), nb::rv_policy::reference_internal)
    .def_prop_ro("midEdges", view(&TriMesh::midEdges), nb::rv_policy::reference_internal)
    .def_prop_ro("minVertexPrincipalDirections", view(&TriMesh::minVertexPrincipalDirections), nb::rv_policy::reference_internal)
    .def_prop_ro("maxVertexPrincipalDirections", view(&TriMesh::maxVertexPrincipalDirections), nb::rv_policy::reference_internal)
    .def_prop_ro("vertexPrincipalCurvatures", view(&TriMesh::vertexPrincipalCurvatures), nb::rv_policy::reference_internal)
    .def_ro("avgEdgeLength", &TriMesh::avgEdgeLength)
    .def_ro("eulerChar", &TriMesh::eulerChar)
    .def_prop_ro("innerEdges", view(&TriMesh::innerEdges), nb::rv_policy::reference_internal)
    .def_ro("numGenerators", &TriMesh::numGenerators)
    .def_ro("boundaryLoops", &TriMesh::boundaryLoops);

    m.def("readOBJ", &read_mesh<directional::readOBJ>, "fileName"_a);
    m.def("readOFF", &read_mesh<directional::readOFF>, "fileName"_a);
    m.def("readDMAT", [](const std::string& fileName){
        Eigen::MatrixXd dmat;
        if (!directional::readDMAT(fileName, dmat))
            throw std::runtime_error("could not read " + fileName);
        return dmat;
    }, "fileName"_a);
    m.def("writeOBJ", [](const std::string& fileName, const TriMesh& mesh, const Eigen::MatrixXd& TC, const Eigen::MatrixXi& FTC,
                         const std::string& mtlFileName, const std::string& textureName){
        require(TC.cols()>=2, "TC must be #TC by 2");
        require(FTC.rows()==mesh.F.rows() && FTC.cols()==3, "FTC must be #F by 3");
        require(FTC.minCoeff()>=0 && FTC.maxCoeff()<TC.rows(), "FTC contains out-of-range indices into TC");
        return directional::writeOBJ(fileName, mesh, TC, FTC, mtlFileName, textureName);
    }, "fileName"_a, "mesh"_a, "TC"_a, "FTC"_a, "mtlFileName"_a="", "textureName"_a="");
}
