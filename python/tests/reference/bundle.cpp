// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: tangent bundle data.
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/IntrinsicVertexTangentBundle.h>
#include <directional/ExtrinsicVertexTangentBundle.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>
#include "write.h"

void write_bundle(const std::string& outDir, const std::string& name, const directional::TangentBundle& tb)
{
    write(outDir, name + "_sources", tb.sources);
    write(outDir, name + "_normals", tb.normals);
    write(outDir, name + "_adjSpaces", tb.adjSpaces);
    write(outDir, name + "_cycleSources", tb.cycleSources);
    write(outDir, name + "_avgAdjLength", tb.avgAdjLength);
    write(outDir, name + "_cycleCurvatures", tb.cycleCurvatures);
    write(outDir, name + "_oneRing", tb.oneRing);
    write(outDir, name + "_innerAdjacencies", tb.innerAdjacencies);
    write(outDir, name + "_connection", (Eigen::MatrixXd(tb.connection.size(), 2) << tb.connection.real(), tb.connection.imag()).finished());
    write_sparse(outDir, name + "_connectionMass", tb.connectionMass);
    write_sparse(outDir, name + "_tangentSpaceMass", tb.tangentSpaceMass);
}

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::IntrinsicVertexTangentBundle vtb;
    directional::ExtrinsicVertexTangentBundle evtb;
    directional::readOBJ(dataDir + "/spherers.obj", mesh);
    ftb.init(mesh);
    vtb.init(mesh);
    evtb.init(mesh);
    write_bundle(outDir, "face", ftb);
    write_bundle(outDir, "vertex", vtb);
    write_bundle(outDir, "extrinsic", evtb);

    //a field on the extrinsic bundle, since its connection is not exposed
    Eigen::VectorXi constVertices(1);
    constVertices << 0;
    Eigen::MatrixXd constVectors(1, 3);
    constVectors << 1.0, 0.0, 0.0;
    directional::CartesianField powerField, rawField;
    directional::power_field(evtb, constVertices, constVectors, Eigen::VectorXd::Constant(1, -1.0), 4, powerField);
    directional::power_to_raw(powerField, 4, rawField);
    write(outDir, "extrinsic_extField", rawField.extField);
}
