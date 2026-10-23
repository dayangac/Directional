// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <directional/setup_mesher.h>
#include <directional/mesher.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::IntegrationData;
using directional::MesherData;
using directional::TriMesh;

/***
 Binds the mesher of seamless N-functions (built only with GMP, which the exact arithmetic uses).
 mesher() returns the polygonal mesh (V, D, F), where row i of F has D(i) vertex indices and is padded with -1.
 ***/
void bind_meshing(nb::module_& m)
{
    nb::class_<MesherData>(m, "MesherData")
    .def_ro("N", &MesherData::N)
    .def_prop_ro("cutV", view(&MesherData::cutV), nb::rv_policy::reference_internal)
    .def_prop_ro("cutF", view(&MesherData::cutF), nb::rv_policy::reference_internal)
    .def_prop_ro("vertexNFunction", view(&MesherData::vertexNFunction), nb::rv_policy::reference_internal)
    .def_prop_ro("integerVars", view(&MesherData::integerVars), nb::rv_policy::reference_internal)
    .def_rw("exactResolution", &MesherData::exactResolution)
    .def_rw("verbose", &MesherData::verbose);

    m.def("setup_mesher", [](const TriMesh& meshCut, const IntegrationData& intData){
        require(intData.vertexTrans2CutMat.rows()==intData.N*meshCut.V.rows() &&
                intData.nVertexFunction.size()==intData.intSpanMat.cols(),
                "intData must be integrated by integrate() with this meshCut");
        MesherData mData;
        directional::setup_mesher(meshCut, intData, mData);
        return mData;
    }, "meshCut"_a, "intData"_a);

    m.def("mesher", [](const TriMesh& origMesh, const MesherData& mData){
        require(mData.cutF.rows()==origMesh.F.rows(), "origMesh must be the whole mesh of the cut mesh in mData");
        Eigen::MatrixXd VOutput;
        Eigen::VectorXi DOutput;
        Eigen::MatrixXi FOutput;
        if (!directional::mesher(origMesh, mData, VOutput, DOutput, FOutput))
            throw std::runtime_error("meshing failed");
        for (int i=0;i<FOutput.rows();i++)
            FOutput.row(i).tail(FOutput.cols()-DOutput(i)).setConstant(-1);
        return std::make_tuple(VOutput, DOutput, FOutput);
    }, "origMesh"_a, "mData"_a);
}
