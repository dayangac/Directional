// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: the combed field and the curl matching of a raw field.
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/read_raw_field.h>
#include <directional/principal_matching.h>
#include <directional/combing.h>
#include <directional/curl_matching.h>
#include "write.h"

void write_matching(const std::string& outDir, const std::string& name, const directional::CartesianField& field)
{
    write(outDir, name + "_extField", field.extField);
    write(outDir, name + "_matching", field.matching);
    write(outDir, name + "_singLocalCycles", field.singLocalCycles);
    write(outDir, name + "_singIndices", field.singIndices);
}

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField rawField, combedField;
    int N;
    directional::readOBJ(dataDir + "/lilium.obj", mesh);
    ftb.init(mesh);
    directional::read_raw_field(dataDir + "/lilium.rawfield", ftb, N, rawField);
    directional::principal_matching(rawField);
    directional::combing(rawField, combedField);
    write_matching(outDir, "combed", combedField);

    Eigen::VectorXd curlNorm;
    directional::curl_matching(rawField, curlNorm);
    write_matching(outDir, "curl", rawField);
    write(outDir, "curl_effort", rawField.effort);
    write(outDir, "curl_curlNorm", curlNorm);
}
