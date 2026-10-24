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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <directional/TriMesh.h>
#include <directional/readOBJ.h>
#include <directional/readOFF.h>
#include <directional/readDMAT.h>
#include <directional/writeOBJ.h>
#include <directional/dual_cycles.h>
#include <directional/barycentric_subdivision.h>
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

//The half-edge accessors of TriMesh take an index, so they are exposed as arrays over all elements.
template <typename Accessor>
Eigen::VectorXi per_element(const int size, const Accessor& accessor)
{
    Eigen::VectorXi values(size);
    for (int i=0;i<size;i++)
        values(i) = accessor(i);
    return values;
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
    .def_ro("boundaryLoops", &TriMesh::boundaryLoops)
    .def_prop_ro("TT", view(&TriMesh::TT), nb::rv_policy::reference_internal)
    .def_prop_ro("EFi", view(&TriMesh::EFi), nb::rv_policy::reference_internal)
    .def_prop_ro("VE", view(&TriMesh::VE), nb::rv_policy::reference_internal)
    .def_prop_ro("VF", view(&TriMesh::VF), nb::rv_policy::reference_internal)
    .def_prop_ro("boundEdges", view(&TriMesh::boundEdges), nb::rv_policy::reference_internal)
    .def_prop_ro("vertexValence", view(&TriMesh::vertexValence), nb::rv_policy::reference_internal)
    .def_prop_ro("isBoundaryVertex", view(&TriMesh::isBoundaryVertex), nb::rv_policy::reference_internal)
    .def_prop_ro("isBoundaryEdge", view(&TriMesh::isBoundaryEdge), nb::rv_policy::reference_internal)
    .def_prop_ro("FBx", view(&TriMesh::FBx), nb::rv_policy::reference_internal)
    .def_prop_ro("FBy", view(&TriMesh::FBy), nb::rv_policy::reference_internal)
    .def_prop_ro("VBx", view(&TriMesh::VBx), nb::rv_policy::reference_internal)
    .def_prop_ro("VBy", view(&TriMesh::VBy), nb::rv_policy::reference_internal)
    .def_prop_ro("GaussianCurvature", view(&TriMesh::GaussianCurvature), nb::rv_policy::reference_internal)
    .def_prop_ro("minBox", view(&TriMesh::minBox), nb::rv_policy::reference_internal)
    .def_prop_ro("maxBox", view(&TriMesh::maxBox), nb::rv_policy::reference_internal)
    .def_prop_ro("VH", [](const TriMesh& mesh){ return per_element(mesh.V.rows(), [&](const int i){ return mesh.VH(i); }); })
    .def_prop_ro("HV", [](const TriMesh& mesh){ return per_element(mesh.dcel.halfedges.size(), [&](const int i){ return mesh.HV(i); }); })
    .def_prop_ro("HE", [](const TriMesh& mesh){ return per_element(mesh.dcel.halfedges.size(), [&](const int i){ return mesh.HE(i); }); })
    .def_prop_ro("HF", [](const TriMesh& mesh){ return per_element(mesh.dcel.halfedges.size(), [&](const int i){ return mesh.HF(i); }); })
    .def_prop_ro("nextH", [](const TriMesh& mesh){ return per_element(mesh.dcel.halfedges.size(), [&](const int i){ return mesh.nextH(i); }); })
    .def_prop_ro("prevH", [](const TriMesh& mesh){ return per_element(mesh.dcel.halfedges.size(), [&](const int i){ return mesh.prevH(i); }); })
    .def_prop_ro("twinH", [](const TriMesh& mesh){ return per_element(mesh.dcel.halfedges.size(), [&](const int i){ return mesh.twinH(i); }); })
    .def_prop_ro("FH", [](const TriMesh& mesh){ return per_element(mesh.F.rows(), [&](const int i){ return mesh.FH(i); }); })
    //#E by 2: the halfedge of each edge, and its twin (-1 on the boundary)
    .def_prop_ro("EH", [](const TriMesh& mesh){
        Eigen::MatrixXi EH(mesh.EV.rows(), 2);
        for (int i=0;i<EH.rows();i++)
            EH.row(i) << mesh.EH(i, 0), mesh.EH(i, 1);
        return EH;
    });

    m.def("readOBJ", &read_mesh<directional::readOBJ>, "fileName"_a);
    m.def("readOFF", &read_mesh<directional::readOFF>, "fileName"_a);
    m.def("readDMAT", [](const std::string& fileName){
        Eigen::MatrixXd dmat;
        if (!directional::readDMAT(fileName, dmat))
            throw std::runtime_error("could not read " + fileName);
        return dmat;
    }, "fileName"_a);
    //Returns (basisCycles, cycleCurvature, vertex2cycle, innerEdges).
    m.def("dual_cycles", [](const TriMesh& mesh){
        Eigen::SparseMatrix<double> basisCycles;
        Eigen::VectorXd cycleCurvature;
        Eigen::VectorXi vertex2cycle, innerEdges;
        directional::dual_cycles(mesh, basisCycles, cycleCurvature, vertex2cycle, innerEdges);
        return std::make_tuple(basisCycles, cycleCurvature, vertex2cycle, innerEdges);
    }, "mesh"_a);

    //Returns (VFine, FFine), with a new vertex at the barycenter of each face.
    m.def("barycentric_subdivision", [](const Eigen::MatrixXd& V, const Eigen::MatrixXi& F){
        require(V.cols()==3 && F.cols()==3, "V and F must be #V by 3 and #F by 3");
        require(F.size()==0 || (F.minCoeff()>=0 && F.maxCoeff()<V.rows()), "F contains out-of-range vertex indices");
        Eigen::MatrixXd VFine;
        Eigen::MatrixXi FFine;
        directional::barycentric_subdivision(V, F, VFine, FFine);
        return std::make_pair(VFine, FFine);
    }, "V"_a, "F"_a);

    m.def("writeOBJ", [](const std::string& fileName, const TriMesh& mesh, const Eigen::MatrixXd& TC, const Eigen::MatrixXi& FTC,
                         const std::string& mtlFileName, const std::string& textureName){
        require(TC.cols()>=2, "TC must be #TC by 2");
        require(FTC.rows()==mesh.F.rows() && FTC.cols()==3, "FTC must be #F by 3");
        require(FTC.minCoeff()>=0 && FTC.maxCoeff()<TC.rows(), "FTC contains out-of-range indices into TC");
        return directional::writeOBJ(fileName, mesh, TC, FTC, mtlFileName, textureName);
    }, "fileName"_a, "mesh"_a, "TC"_a, "FTC"_a, "mtlFileName"_a="", "textureName"_a="");
}
