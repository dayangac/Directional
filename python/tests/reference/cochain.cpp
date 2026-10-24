// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: exact projections, a Hodge decomposition, and the harmonic projection of a 1-form on a torus.
#include <directional/readOBJ.h>
#include <directional/discrete_exterior_calculus.h>
#include <directional/cochain_complex.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::readOBJ(dataDir + "/torus.obj", mesh);
    Eigen::SparseMatrix<double> d0 = directional::d0_matrix<double>(mesh), d1 = directional::d1_matrix<double>(mesh);
    Eigen::SparseMatrix<double> h1, invh1, h2, invh2;
    directional::hodge_star_1_matrix(mesh, h1, invh1);
    directional::hodge_star_2_matrix<double>(mesh, h2, invh2);

    //a 1-form that is neither exact nor coexact
    Eigen::VectorXd oneForm = mesh.midEdges.col(0).array().sin()+mesh.midEdges.col(1).array().cos();
    Eigen::VectorXd prevCochain, exactCochain;
    directional::project_exact<double>(d0, h1, oneForm, prevCochain, exactCochain);
    write(outDir, "exact_prevCochain", prevCochain);
    write(outDir, "exact_exactCochain", exactCochain);
    Eigen::VectorXd curl = d1*oneForm;
    directional::project_exact<double>(d1, h1, curl, prevCochain, exactCochain, true);
    write(outDir, "gauge_prevCochain", prevCochain);
    write(outDir, "gauge_exactCochain", exactCochain);

    Eigen::VectorXd coexactCochain, harmCochain, nextCochain;
    directional::hodge_decomposition<double>(d0, d1, h1, h2, oneForm, 1, exactCochain, coexactCochain, harmCochain, prevCochain, nextCochain);
    write(outDir, "hodge_exactCochain", exactCochain);
    write(outDir, "hodge_coexactCochain", coexactCochain);
    write(outDir, "hodge_harmCochain", harmCochain);

    directional::project_coexact<double>(d1, h1, h2, oneForm, 1, coexactCochain, nextCochain);
    write(outDir, "coexact_coexactCochain", coexactCochain);
    write(outDir, "coexact_nextCochain", nextCochain);

    //the basis is only defined up to a rotation, so the projection onto it is compared
    Eigen::MatrixXd harmBasis;
    directional::cohomology_basis<double>(d0, d1, h1, 2, harmBasis);
    write(outDir, "harmonic_projection", Eigen::VectorXd(harmBasis*(harmBasis.transpose()*h1*oneForm)));
}
