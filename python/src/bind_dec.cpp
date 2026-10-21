// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <tuple>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <directional/TriMesh.h>
#include <directional/discrete_exterior_calculus.h>
#include <directional/cochain_complex.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::TriMesh;
typedef Eigen::SparseMatrix<double> SparseMatrix;

//The metric M must be square and act on a space of the given size.
inline void require_metric(const SparseMatrix& M, const Eigen::Index size, const char* message)
{
    require(M.rows()==size && M.cols()==size, message);
}

/***
 Binds the discrete exterior calculus operators and the cochain-complex functions (for double), which take and return scipy.sparse matrices.
 The isDirichlet arguments are left out, since only the Neumann boundary conditions are implemented in C++.
 The sizes are checked first, since the C++ functions abort on a failed factorization.
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

    //Returns (prevCochain, exactCochain). M is the metric of cochain, or of prevCochain with gauge fixing.
    m.def("project_exact", [](const SparseMatrix& d, const SparseMatrix& M, const Eigen::VectorXd& cochain, const bool gaugeFixing){
        require(cochain.size()==d.rows(), "cochain must have a value per row of d");
        require_metric(M, (gaugeFixing ? d.cols() : d.rows()), "M must be square, of the size of prevCochain with gauge fixing and of cochain without");
        Eigen::VectorXd prevCochain, exactCochain;
        directional::project_exact<double>(d, M, cochain, prevCochain, exactCochain, gaugeFixing);
        return std::make_pair(prevCochain, exactCochain);
    }, "d"_a, "M"_a, "cochain"_a, "gaugeFixing"_a=false);

    //Returns (exactCochain, coexactCochain, harmCochain, prevCochain, nextCochain).
    m.def("hodge_decomposition", [](const SparseMatrix& d, const SparseMatrix& dNext, const SparseMatrix& M, const SparseMatrix& MNext,
                                    const Eigen::VectorXd& cochain, const int k, const bool gaugeFixing){
        require(cochain.size()==d.rows() && dNext.cols()==d.rows(), "cochain must have a value per row of d and per column of dNext");
        require_metric(M, cochain.size(), "M must be square, of the size of cochain");
        require_metric(MNext, dNext.rows(), "MNext must be square, of the size of the rows of dNext");
        Eigen::VectorXd exactCochain, coexactCochain, harmCochain, prevCochain, nextCochain;
        directional::hodge_decomposition<double>(d, dNext, M, MNext, cochain, k, exactCochain, coexactCochain, harmCochain, prevCochain, nextCochain, gaugeFixing);
        return std::make_tuple(exactCochain, coexactCochain, harmCochain, prevCochain, nextCochain);
    }, "d"_a, "dNext"_a, "M"_a, "MNext"_a, "cochain"_a, "k"_a, "gaugeFixing"_a=false);

    m.def("cohomology_basis", [](const SparseMatrix& d, const SparseMatrix& dNext, const SparseMatrix& M, const int bettiNumber){
        require(dNext.cols()==d.rows(), "d and dNext must be consecutive differentials");
        require_metric(M, d.rows(), "M must be square, of the size of the rows of d");
        require(bettiNumber>0 && bettiNumber<=d.rows(), "bettiNumber must be positive and at most the size of the cochains");
        Eigen::MatrixXd harmFields;
        directional::cohomology_basis<double>(d, dNext, M, bettiNumber, harmFields);
        return harmFields;
    }, "d"_a, "dNext"_a, "M"_a, "bettiNumber"_a);
}
