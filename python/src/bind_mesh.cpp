// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <directional/TriMesh.h>
#include <directional/readOBJ.h>
#include <directional/readOFF.h>
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
    .def_prop_ro("faceNormals", view(&TriMesh::faceNormals), nb::rv_policy::reference_internal)
    .def_prop_ro("vertexNormals", view(&TriMesh::vertexNormals), nb::rv_policy::reference_internal)
    .def_prop_ro("faceAreas", view(&TriMesh::faceAreas), nb::rv_policy::reference_internal)
    .def_prop_ro("barycenters", view(&TriMesh::barycenters), nb::rv_policy::reference_internal)
    .def_ro("avgEdgeLength", &TriMesh::avgEdgeLength)
    .def_ro("eulerChar", &TriMesh::eulerChar);

    m.def("readOBJ", &read_mesh<directional::readOBJ>, "fileName"_a);
    m.def("readOFF", &read_mesh<directional::readOFF>, "fileName"_a);
}
