// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: the face-based FEM matrices, on a closed and on an open mesh.
#include <directional/readOBJ.h>
#include <directional/gradient_matrices.h>
#include <directional/curl_matrices.h>
#include <directional/div_matrices.h>
#include <directional/mass_matrices.h>
#include <directional/extrinsic_intrinsic_matrices.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    for (const std::string name : {"spherers", "lilium"}){
        directional::TriMesh mesh;
        directional::readOBJ(dataDir + "/" + name + ".obj", mesh);
        for (const bool isIntrinsic : {false, true}){
            const std::string prefix = name + (isIntrinsic ? "_intrinsic_" : "_extrinsic_");
            write_sparse(outDir, prefix + "conf_gradient", directional::conf_gradient_matrix_2D<double>(mesh, isIntrinsic, 2));
            write_sparse(outDir, prefix + "non_conf_gradient", directional::non_conf_gradient_matrix_2D<double>(mesh, isIntrinsic, 2));
            write_sparse(outDir, prefix + "curl", directional::curl_matrix_2D<double>(mesh, isIntrinsic, 2));
            write_sparse(outDir, prefix + "div", directional::div_matrix_2D<double>(mesh, isIntrinsic, 2));
            write_sparse(outDir, prefix + "rotation", directional::face_vector_rotation_matrix_2D<double>(mesh, isIntrinsic, 2));
        }
        write_sparse(outDir, name + "_face_mass", directional::face_mass_matrix_2D<double>(mesh, false, 2));
        write_sparse(outDir, name + "_inverse_face_mass", directional::face_mass_matrix_2D<double>(mesh, true));
        write_sparse(outDir, name + "_edge_diamond_mass", directional::edge_diamond_mass_matrix_2D<double>(mesh));
        write_sparse(outDir, name + "_conf_mass", directional::conf_mass_matrix_2D<double>(mesh, 3));
        write_sparse(outDir, name + "_lumped_voronoi_mass", directional::lumped_voronoi_mass_matrix_2D<double>(mesh));
        write_sparse(outDir, name + "_extrinsic_to_intrinsic", directional::face_extrinsic_to_intrinsic_matrix_2D<double>(mesh, 2));
        write_sparse(outDir, name + "_intrinsic_to_extrinsic", directional::face_intrinsic_to_extrinsic_matrix_2D<double>(mesh));
    }
}
