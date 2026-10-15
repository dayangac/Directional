// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: singularities read from a file.
#include <directional/readOFF.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/read_raw_field.h>
#include <directional/read_singularities.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField field;
    int N;
    directional::readOFF(dataDir + "/bumpy.off", mesh);
    ftb.init(mesh);
    directional::read_raw_field(dataDir + "/bumpy.rawfield", ftb, N, field);
    directional::read_singularities(dataDir + "/bumpy.sings", field);
    write(outDir, "singLocalCycles", field.singLocalCycles);
    write(outDir, "singIndices", field.singIndices);
}
