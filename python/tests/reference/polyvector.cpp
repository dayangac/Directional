// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

//Reference for test_reference.py: PolyVector fields, with and without iterations.
#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/read_raw_field.h>
#include <directional/polyvector_field.h>
#include <directional/polyvector_iteration_functions.h>
#include <directional/polyvector_to_raw.h>
#include <directional/raw_to_polyvector.h>
#include "write.h"

//Two edge directions on face 0, and one on faces 300 and 900
void set_constraints(const directional::TriMesh& mesh, directional::PolyVectorData& pvData)
{
    pvData.constSpaces.resize(4);
    pvData.constSpaces << 0, 0, 300, 900;
    Eigen::MatrixXi edges(4, 2);
    edges << 0, 1, 0, 2, 0, 1, 1, 2;
    pvData.constVectors.resize(4, 3);
    for (int i=0;i<4;i++)
        pvData.constVectors.row(i) = (mesh.V.row(mesh.F(pvData.constSpaces(i), edges(i, 1)))-mesh.V.row(mesh.F(pvData.constSpaces(i), edges(i, 0)))).normalized();
    pvData.wAlignment = Eigen::VectorXd::Constant(4, 1.0);
}

void write_polyvector(const std::string& outDir, const std::string& name, const directional::CartesianField& pvField)
{
    directional::CartesianField rawField;
    directional::polyvector_to_raw(pvField, rawField);
    write(outDir, name + "_intField", pvField.intField);
    write(outDir, name + "_extField", rawField.extField);
}

int main(int argc, char** argv)
{
    const std::string dataDir = argv[1], outDir = argv[2];
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::readOBJ(dataDir + "/spherers.obj", mesh);
    ftb.init(mesh);

    for (const double alignWeight : {-1.0, 1.0}){
        directional::PolyVectorData pvData;
        directional::CartesianField pvField;
        pvData.N = 4;
        pvData.tb = &ftb;
        set_constraints(mesh, pvData);
        pvData.wAlignment.setConstant(alignWeight);
        directional::polyvector_field(pvData, pvField);
        write_polyvector(outDir, (alignWeight<0.0 ? "hard" : "soft"), pvField);
    }

    //the Ginzburg-Landau iterations of tutorial 303, and the curl-reducing iterations of tutorial 304
    for (const std::string name : {"hard_rosy", "curl_projection"}){
        directional::PolyVectorData pvData;
        directional::CartesianField pvField;
        pvData.N = 4;
        pvData.tb = &ftb;
        set_constraints(mesh, pvData);
        if (name=="hard_rosy"){
            pvData.wRoSy = -1.0;
            pvData.initImplicitFactor = 0.2;
            pvData.implicitScheduler = 1.0;
        } else
            pvData.wRoSy = 5.0;
        pvData.iterationMode = true;
        directional::polyvector_field(pvData, pvField);
        if (name=="hard_rosy")
            directional::polyvector_iterate(pvData, pvField, {directional::hard_rosy}, 5);
        else
            directional::polyvector_iterate(pvData, pvField, {directional::soft_rosy, directional::curl_projection}, 3);
        write_polyvector(outDir, name, pvField);
    }

    directional::TriMesh lilium;
    directional::PCFaceTangentBundle liliumFtb;
    directional::CartesianField rawField, pvField;
    int N;
    directional::readOBJ(dataDir + "/lilium.obj", lilium);
    liliumFtb.init(lilium);
    directional::read_raw_field(dataDir + "/lilium.rawfield", liliumFtb, N, rawField);
    directional::raw_to_polyvector(rawField, pvField);
    write(outDir, "raw_to_polyvector_intField", pvField.intField);
}
