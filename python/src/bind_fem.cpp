// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <directional/TriMesh.h>
#include <directional/gradient_matrices.h>
#include <directional/curl_matrices.h>
#include <directional/div_matrices.h>
#include <directional/mass_matrices.h>
#include <directional/extrinsic_intrinsic_matrices.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::TriMesh;
typedef Eigen::SparseMatrix<double> SparseMatrix;

/***
 Binds the face-based FEM matrices (for double), which return scipy.sparse matrices.
 The polynomial degree arguments (d, deg) are left out, since the C++ functions only implement their default value.
 ***/
void bind_fem(nb::module_& m)
{
    m.def("conf_gradient_matrix_2D", [](const TriMesh& mesh, const bool isIntrinsic, const int N){
        require(N>0, "N must be positive");
        return directional::conf_gradient_matrix_2D<double>(mesh, isIntrinsic, N);
    }, "mesh"_a, "isIntrinsic"_a=false, "N"_a=1);

    m.def("non_conf_gradient_matrix_2D", [](const TriMesh& mesh, const bool isIntrinsic, const int N){
        require(N>0, "N must be positive");
        return directional::non_conf_gradient_matrix_2D<double>(mesh, isIntrinsic, N);
    }, "mesh"_a, "isIntrinsic"_a=false, "N"_a=1);

    m.def("curl_matrix_2D", [](const TriMesh& mesh, const bool isIntrinsic, const int N, const std::optional<Eigen::VectorXi>& matching){
        require(N>0, "N must be positive");
        require(!matching || matching->size()==mesh.EV.rows(), "matching must have a value per edge");
        return directional::curl_matrix_2D<double>(mesh, isIntrinsic, N, 1, matching.value_or(Eigen::VectorXi()));
    }, "mesh"_a, "isIntrinsic"_a=false, "N"_a=1, "matching"_a=nb::none());

    m.def("div_matrix_2D", [](const TriMesh& mesh, const bool isIntrinsic, const int N){
        require(N>0, "N must be positive");
        return directional::div_matrix_2D<double>(mesh, isIntrinsic, N);
    }, "mesh"_a, "isIntrinsic"_a=false, "N"_a=1);

    m.def("face_vector_rotation_matrix_2D", [](const TriMesh& mesh, const bool isIntrinsic, const int N){
        require(N>0, "N must be positive");
        return directional::face_vector_rotation_matrix_2D<double>(mesh, isIntrinsic, N);
    }, "mesh"_a, "isIntrinsic"_a=false, "N"_a=1);

    m.def("conf_mass_matrix_2D", [](const TriMesh& mesh, const int dim){
        require(dim>0, "dim must be positive");
        return directional::conf_mass_matrix_2D<double>(mesh, dim);
    }, "mesh"_a, "dim"_a=1);

    m.def("lumped_voronoi_mass_matrix_2D", [](const TriMesh& mesh, const int dim){
        require(dim>0, "dim must be positive");
        return directional::lumped_voronoi_mass_matrix_2D<double>(mesh, dim);
    }, "mesh"_a, "dim"_a=1);

    m.def("face_mass_matrix_2D", [](const TriMesh& mesh, const bool isInverse, const int dim){
        require(dim>0, "dim must be positive");
        return directional::face_mass_matrix_2D<double>(mesh, isInverse, dim);
    }, "mesh"_a, "isInverse"_a=false, "dim"_a=3);

    m.def("edge_diamond_mass_matrix_2D", [](const TriMesh& mesh, const bool isInverse){
        return directional::edge_diamond_mass_matrix_2D<double>(mesh, isInverse);
    }, "mesh"_a, "isInverse"_a=false);

    m.def("face_extrinsic_to_intrinsic_matrix_2D", [](const TriMesh& mesh, const int N){
        require(N>0, "N must be positive");
        return directional::face_extrinsic_to_intrinsic_matrix_2D<double>(mesh, N);
    }, "mesh"_a, "N"_a=1);

    m.def("face_intrinsic_to_extrinsic_matrix_2D", [](const TriMesh& mesh, const int N){
        require(N>0, "N must be positive");
        return directional::face_intrinsic_to_extrinsic_matrix_2D<double>(mesh, N);
    }, "mesh"_a, "N"_a=1);
}
