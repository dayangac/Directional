// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: a matrix read from a DMAT file.
#include <directional/readDMAT.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    Eigen::MatrixXd dmat;
    directional::readDMAT(dataDir + "/spheres_constFaces.dmat", dmat);
    write(outDir, "dmat", dmat);
}
