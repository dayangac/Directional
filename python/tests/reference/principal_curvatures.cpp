// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: vertex principal directions and curvatures.
#include <directional/readOBJ.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::readOBJ(dataDir + "/torus.obj", mesh);
    write(outDir, "minVertexPrincipalDirections", mesh.minVertexPrincipalDirections);
    write(outDir, "maxVertexPrincipalDirections", mesh.maxVertexPrincipalDirections);
    write(outDir, "vertexPrincipalCurvatures", mesh.vertexPrincipalCurvatures);
}
