// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: the mesh, and hard and soft power fields on face and vertex bundles.
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/IntrinsicVertexTangentBundle.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>
#include <directional/principal_matching.h>
#include <directional/write_raw_field.h>
#include "write.h"

void write_fields(const directional::TangentBundle& tb, const std::string& bundleName, const directional::TriMesh& mesh, const std::string& outDir)
{
    Eigen::VectorXi constSpaces(3);
    constSpaces << 0, 100, 500;
    Eigen::MatrixXd constVectors(3, 3);
    for (int i=0;i<3;i++)
        constVectors.row(i) = mesh.V.row(mesh.F(constSpaces(i), 1))-mesh.V.row(mesh.F(constSpaces(i), 0));

    for (const double alignWeight : {-1.0, 1.0}){
        directional::CartesianField powerField, rawField;
        directional::power_field(tb, constSpaces, constVectors, Eigen::VectorXd::Constant(3, alignWeight), 4, powerField);
        directional::power_to_raw(powerField, 4, rawField);
        directional::principal_matching(rawField);
        const std::string name = bundleName + (alignWeight<0.0 ? "_hard" : "_soft");
        write(outDir, name + "_extField", rawField.extField);
        write(outDir, name + "_matching", rawField.matching);
        write(outDir, name + "_effort", rawField.effort);
        write(outDir, name + "_singLocalCycles", rawField.singLocalCycles);
        write(outDir, name + "_singIndices", rawField.singIndices);
        directional::write_raw_field(outDir + "/" + name + ".rawfield", rawField, true);
    }
}

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::readOBJ(dataDir + "/spherers.obj", mesh);
    write(outDir, "EV", mesh.EV);
    write(outDir, "EF", mesh.EF);
    write(outDir, "FE", mesh.FE);
    write(outDir, "faceNormals", mesh.faceNormals);
    write(outDir, "vertexNormals", mesh.vertexNormals);
    write(outDir, "faceAreas", mesh.faceAreas);

    directional::PCFaceTangentBundle ftb;
    directional::IntrinsicVertexTangentBundle vtb;
    ftb.init(mesh);
    vtb.init(mesh);
    write_fields(ftb, "face", mesh, outDir);
    write_fields(vtb, "vertex", mesh, outDir);
}
