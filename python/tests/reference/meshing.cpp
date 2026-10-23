// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: the polygonal mesh of the integer isolines of a seamless 4-function, as in tutorial 505.
#include <directional/readOFF.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/read_raw_field.h>
#include <directional/principal_matching.h>
#include <directional/setup_integration.h>
#include <directional/integrate.h>
#include <directional/setup_mesher.h>
#include <directional/mesher.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh meshWhole, meshCut;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField rawField, combedField;
    int N;
    directional::readOFF(dataDir + "/vase.off", meshWhole);
    ftb.init(meshWhole);
    directional::read_raw_field(dataDir + "/vase-4.rawfield", ftb, N, rawField);
    directional::principal_matching(rawField);

    directional::IntegrationData intData(N);
    directional::setup_integration(rawField, intData, meshCut, combedField);
    intData.integralSeamless = true;
    intData.roundSeams = false;
    Eigen::MatrixXd NFunction, NCornerFunctions;
    directional::integrate(combedField, intData, meshCut, NFunction, NCornerFunctions);

    directional::MesherData mData;
    directional::setup_mesher(meshCut, intData, mData);
    Eigen::MatrixXd VPolyMesh;
    Eigen::VectorXi DPolyMesh;
    Eigen::MatrixXi FPolyMesh;
    write(outDir, "success", directional::mesher(meshWhole, mData, VPolyMesh, DPolyMesh, FPolyMesh));
    write(outDir, "VPolyMesh", VPolyMesh);
    write(outDir, "DPolyMesh", DPolyMesh);
    write(outDir, "FPolyMesh", FPolyMesh);
}
