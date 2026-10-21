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
#include <Eigen/Sparse>

//Writes a matrix in full precision as outDir/name.txt, which the tests load with numpy.loadtxt().
template <typename Matrix>
void write(const std::string& outDir, const std::string& name, const Matrix& matrix)
{
    std::ofstream(outDir + "/" + name + ".txt") << std::setprecision(17) << matrix << std::endl;
}

//Writes a sparse matrix as its shape (outDir/name_shape.txt) and its nonzeros, one (row, column, value) per line (outDir/name.txt).
inline void write_sparse(const std::string& outDir, const std::string& name, const Eigen::SparseMatrix<double>& matrix)
{
    Eigen::MatrixXd entries(matrix.nonZeros(), 3);
    int entry = 0;
    for (int k=0;k<matrix.outerSize();k++)
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it)
            entries.row(entry++) << it.row(), it.col(), it.value();
    write(outDir, name, entries);
    write(outDir, name + "_shape", Eigen::RowVector2i(matrix.rows(), matrix.cols()));
}

#endif
