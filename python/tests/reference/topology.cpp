// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: mesh topology and the cycles of the face bundle, on a closed and on an open mesh,
//and the field of tutorial 401, with the singularities of a file prescribed on their cycles.
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/read_singularities.h>
#include <directional/index_prescription.h>
#include "write.h"

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    for (const std::string name : {"fertility", "lilium"}){
        directional::TriMesh mesh;
        directional::PCFaceTangentBundle ftb;
        directional::readOBJ(dataDir + "/" + name + ".obj", mesh);
        ftb.init(mesh);
        write(outDir, name + "_innerEdges", mesh.innerEdges);
        write(outDir, name + "_numGenerators", mesh.numGenerators);
        write(outDir, name + "_numBoundaryLoops", mesh.boundaryLoops.size());
        for (int i=0;i<mesh.boundaryLoops.size();i++)
            write(outDir, name + "_boundaryLoop" + std::to_string(i), Eigen::Map<const Eigen::VectorXi>(mesh.boundaryLoops[i].data(), mesh.boundaryLoops[i].size()));
        write(outDir, name + "_local2Cycle", ftb.local2Cycle);

        //the nonzeros of the cycles, one (row, column, value) per line
        Eigen::MatrixXd cycleEntries(ftb.cycles.nonZeros(), 3);
        int entry = 0;
        for (int k=0;k<ftb.cycles.outerSize();k++)
            for (Eigen::SparseMatrix<double>::InnerIterator it(ftb.cycles, k); it; ++it)
                cycleEntries.row(entry++) << it.row(), it.col(), it.value();
        write(outDir, name + "_cycles", cycleEntries);
    }

    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField field;
    directional::readOBJ(dataDir + "/fertility.obj", mesh);
    ftb.init(mesh);
    field.init(ftb, directional::fieldTypeEnum::RAW_FIELD, 2);
    Eigen::VectorXi singVertices, singIndices;
    int N;
    directional::read_singularities(dataDir + "/fertility.sings", N, singVertices, singIndices);
    Eigen::VectorXi cycleIndices = Eigen::VectorXi::Zero(ftb.cycles.rows());
    for (int i=0;i<singVertices.size();i++)
        cycleIndices(ftb.local2Cycle(singVertices(i))) = singIndices(i);
    Eigen::VectorXd rotationAngles;
    double linfError;
    directional::index_prescription(cycleIndices, 2, 0.0, field, rotationAngles, linfError);
    write(outDir, "prescribed_extField", field.extField);
    write(outDir, "prescribed_linfError", linfError);
}
