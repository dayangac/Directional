// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: seamless integration rounding the seams and then the singularities, as in tutorial 503.
#include <directional/readOFF.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/read_raw_field.h>
#include <directional/principal_matching.h>
#include <directional/setup_integration.h>
#include <directional/integrate.h>
#include <directional/branched_isolines.h>
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
    intData.integralSeamless = true;
    for (const bool roundSeams : {true, false}){
        const std::string name = (roundSeams ? "seams" : "sings");
        intData.roundSeams = roundSeams;
        directional::setup_integration(rawField, intData, meshCut, combedField);
        Eigen::MatrixXd NFunction, NCornerFunctions;
        directional::integrate(combedField, intData, meshCut, NFunction, NCornerFunctions);
        write(outDir, name + "_cutV", meshCut.V);
        write(outDir, name + "_cutF", meshCut.F);
        write(outDir, name + "_face2cut", intData.face2cut);
        write(outDir, name + "_constrainedVertices", intData.constrainedVertices);
        write(outDir, name + "_matching", combedField.matching);
        write(outDir, name + "_extField", combedField.extField);
        write(outDir, name + "_NFunction", NFunction);
        write(outDir, name + "_NCornerFunctions", NCornerFunctions);
        write(outDir, name + "_nVertexFunction", intData.nVertexFunction);

        Eigen::MatrixXd isoV, isoN;
        Eigen::MatrixXi isoE, isoOrigE;
        Eigen::VectorXi funcNum;
        directional::branched_isolines(meshCut.V, meshCut.F, NFunction, isoV, isoE, isoOrigE, isoN, funcNum);
        write(outDir, name + "_isoV", isoV);
        write(outDir, name + "_isoE", isoE);
        write(outDir, name + "_funcNum", funcNum);
    }
}
