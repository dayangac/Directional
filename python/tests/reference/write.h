// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DIRECTIONAL_PYTHON_REFERENCE_WRITE_H
#define DIRECTIONAL_PYTHON_REFERENCE_WRITE_H

#include <fstream>
#include <iomanip>
#include <string>
#include <Eigen/Core>

//Writes a matrix in full precision as outDir/name.txt, which the tests load with numpy.loadtxt().
template <typename Matrix>
void write(const std::string& outDir, const std::string& name, const Matrix& matrix)
{
    std::ofstream(outDir + "/" + name + ".txt") << std::setprecision(17) << matrix << std::endl;
}

#endif
