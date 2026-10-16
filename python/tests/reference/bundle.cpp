// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: tangent bundle data.
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/IntrinsicVertexTangentBundle.h>
#include "write.h"

void write_bundle(const std::string& outDir, const std::string& name, const directional::TangentBundle& tb)
{
    write(outDir, name + "_sources", tb.sources);
    write(outDir, name + "_normals", tb.normals);
    write(outDir, name + "_cycleSources", tb.cycleSources);
    write(outDir, name + "_avgAdjLength", tb.avgAdjLength);
}

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::IntrinsicVertexTangentBundle vtb;
    directional::readOBJ(dataDir + "/spherers.obj", mesh);
    ftb.init(mesh);
    vtb.init(mesh);
    write_bundle(outDir, "face", ftb);
    write_bundle(outDir, "vertex", vtb);
}
