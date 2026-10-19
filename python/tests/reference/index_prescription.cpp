// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: a field with prescribed singularities.
#include <numbers>
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/index_prescription.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField field;
    directional::readOBJ(dataDir + "/spherers.obj", mesh);
    ftb.init(mesh);
    field.init(ftb, directional::fieldTypeEnum::RAW_FIELD, 2);
    Eigen::VectorXi cycleIndices = Eigen::VectorXi::Zero(ftb.cycles.rows());
    cycleIndices(35) = cycleIndices(36) = 2;
    Eigen::VectorXd rotationAngles;
    double linfError;
    directional::index_prescription(cycleIndices, 2, std::numbers::pi/16.0, field, rotationAngles, linfError);
    write(outDir, "extField", field.extField);
    write(outDir, "rotationAngles", rotationAngles);
    write(outDir, "linfError", linfError);
}
