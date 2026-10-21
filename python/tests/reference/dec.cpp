// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: the discrete exterior calculus operators, on a closed and on an open mesh.
#include <directional/readOBJ.h>
#include <directional/discrete_exterior_calculus.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    for (const std::string name : {"spherers", "lilium"}){
        directional::TriMesh mesh;
        directional::readOBJ(dataDir + "/" + name + ".obj", mesh);
        write_sparse(outDir, name + "_d0", directional::d0_matrix<double>(mesh));
        write_sparse(outDir, name + "_d1", directional::d1_matrix<double>(mesh));
        Eigen::SparseMatrix<double> M1, h1, invh1, h2, invh2;
        directional::linear_whitney_mass_matrix(mesh, M1);
        write_sparse(outDir, name + "_whitney_mass", M1);
        for (const bool circumcenter : {true, false}){
            directional::hodge_star_1_matrix(mesh, h1, invh1, circumcenter);
            write_sparse(outDir, name + (circumcenter ? "_circumcenter" : "_barycenter") + "_hodge_star_1", h1);
            write_sparse(outDir, name + (circumcenter ? "_circumcenter" : "_barycenter") + "_inv_hodge_star_1", invh1);
        }
        directional::hodge_star_2_matrix<double>(mesh, h2, invh2);
        write_sparse(outDir, name + "_hodge_star_2", h2);
        write_sparse(outDir, name + "_inv_hodge_star_2", invh2);
    }
}
