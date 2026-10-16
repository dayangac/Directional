// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: streamlines traced from given seeds (automatic seeds are random on every run).
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>
#include <directional/streamlines.h>
#include "write.h"

void write_rows(const std::string& outDir, const std::string& name, const std::vector<Eigen::RowVector3d>& rows)
{
    Eigen::MatrixXd matrix(rows.size(), 3);
    for (int i=0;i<rows.size();i++)
        matrix.row(i) = rows[i];
    write(outDir, name, matrix);
}

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField powerField, rawField;
    directional::readOBJ(dataDir + "/spherers.obj", mesh);
    ftb.init(mesh);
    Eigen::VectorXi constFaces(1);
    constFaces << 0;
    Eigen::MatrixXd constVectors = mesh.V.row(mesh.F(0, 1))-mesh.V.row(mesh.F(0, 0));
    directional::power_field(ftb, constFaces, constVectors, Eigen::VectorXd::Constant(1, -1.0), 3, powerField);
    directional::power_to_raw(powerField, 3, rawField, true);

    Eigen::VectorXi seeds(3);
    seeds << 0, 10, 500;
    directional::StreamlineData data;
    directional::StreamlineState state;
    directional::streamlines_init(rawField, seeds, 3.0, data, state);
    for (int i=0;i<3;i++)
        directional::streamlines_next(data, state, 0.5*mesh.avgEdgeLength);
    write(outDir, "samplePoints", data.samplePoints);
    write_rows(outDir, "segStart", state.segStart);
    write_rows(outDir, "segEnd", state.segEnd);
}
