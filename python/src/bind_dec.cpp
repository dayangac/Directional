// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/pair.h>
#include <directional/TriMesh.h>
#include <directional/discrete_exterior_calculus.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::TriMesh;
typedef Eigen::SparseMatrix<double> SparseMatrix;

/***
 Binds the discrete exterior calculus operators (for double), which return scipy.sparse matrices.
 The isDirichlet arguments are left out, since only the Neumann boundary conditions are implemented in C++.
 ***/
void bind_dec(nb::module_& m)
{
    m.def("d0_matrix", [](const TriMesh& mesh){ return directional::d0_matrix<double>(mesh); }, "mesh"_a);
    m.def("d1_matrix", [](const TriMesh& mesh){ return directional::d1_matrix<double>(mesh); }, "mesh"_a);

    m.def("linear_whitney_mass_matrix", [](const TriMesh& mesh){
        SparseMatrix M1;
        directional::linear_whitney_mass_matrix(mesh, M1);
        return M1;
    }, "mesh"_a);

    //Both return (hodgeStar, invHodgeStar).
    m.def("hodge_star_1_matrix", [](const TriMesh& mesh, const bool circumcenter){
        SparseMatrix hodgeStar, invHodgeStar;
        directional::hodge_star_1_matrix(mesh, hodgeStar, invHodgeStar, circumcenter);
        return std::make_pair(hodgeStar, invHodgeStar);
    }, "mesh"_a, "circumcenter"_a=true);

    m.def("hodge_star_2_matrix", [](const TriMesh& mesh){
        SparseMatrix hodgeStar, invHodgeStar;
        directional::hodge_star_2_matrix<double>(mesh, hodgeStar, invHodgeStar);
        return std::make_pair(hodgeStar, invHodgeStar);
    }, "mesh"_a);
}
