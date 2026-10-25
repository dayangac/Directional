// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <directional/streamlines.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::CartesianField;
using directional::StreamlineData;
using directional::StreamlineState;
using directional::discTangTypeEnum;
using directional::fieldTypeEnum;

//Copies a list of points (as stored for the traced segments) into a #points by 3 matrix.
inline Eigen::MatrixXd stack_rows(const std::vector<Eigen::RowVector3d>& rows)
{
    Eigen::MatrixXd stacked(rows.size(), 3);
    for (int i=0;i<rows.size();i++)
        stacked.row(i)=rows[i];
    return stacked;
}

//Copies a list of values (as stored for the traced segments) into a vector.
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> to_vector(const std::vector<T>& list)
{
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(list.data(), list.size());
}

/***
 Binds streamline tracing. Unlike the other bindings, streamlines_init() fills the given data and state in place, as in C++:
 StreamlineData holds a field that can point to a tangent bundle stored inside the same StreamlineData, so it must never be copied or moved.
 The Python constructors value-initialize the structs, so that using them before streamlines_init() can be detected (null slMesh).
 ***/
void bind_streamlines(nb::module_& m)
{
    nb::class_<StreamlineData>(m, "StreamlineData")
    .def("__init__", [](StreamlineData* data){ new (data) StreamlineData(); })
    .def_prop_ro("sampleFaces", view(&StreamlineData::sampleFaces), nb::rv_policy::reference_internal)
    .def_prop_ro("samplePoints", view(&StreamlineData::samplePoints), nb::rv_policy::reference_internal);

    nb::class_<StreamlineState>(m, "StreamlineState")
    .def("__init__", [](StreamlineState* state){ new (state) StreamlineState(); })
    .def_ro("currTime", &StreamlineState::currTime)
    .def_prop_ro("segmentAlive", view(&StreamlineState::segmentAlive), nb::rv_policy::reference_internal)
    .def_prop_ro("segStart", [](const StreamlineState& state){ return stack_rows(state.segStart); })
    .def_prop_ro("segEnd", [](const StreamlineState& state){ return stack_rows(state.segEnd); })
    .def_prop_ro("segNormal", [](const StreamlineState& state){ return stack_rows(state.segNormal); })
    .def_prop_ro("segOrigFace", [](const StreamlineState& state){ return to_vector(state.segOrigFace); })
    .def_prop_ro("segOrigVector", [](const StreamlineState& state){ return to_vector(state.segOrigVector); })
    .def_prop_ro("segTimeSignatures", [](const StreamlineState& state){ return to_vector(state.segTimeSignatures); });

    //data keeps the field alive (keep_alive), since it points to the field's tangent bundle and mesh.
    m.def("streamlines_init", [](const CartesianField& field,
                                 const Eigen::VectorXi& seedLocations,
                                 const double distRatio,
                                 StreamlineData& data,
                                 StreamlineState& state){
        require(field.fieldType==fieldTypeEnum::RAW_FIELD, "field must be a RAW_FIELD");
        //the library casts vertex bundles to IntrinsicVertexTangentBundle, so other bundles are rejected
        const directional::PCFaceTangentBundle* ftb = dynamic_cast<const directional::PCFaceTangentBundle*>(field.tb);
        const directional::IntrinsicVertexTangentBundle* vtb = dynamic_cast<const directional::IntrinsicVertexTangentBundle*>(field.tb);
        require(ftb!=nullptr || vtb!=nullptr, "field must be on a PCFaceTangentBundle or an IntrinsicVertexTangentBundle");
        const directional::TriMesh* mesh = (ftb!=nullptr ? ftb->mesh : vtb->mesh);
        require(seedLocations.size()==0 || (seedLocations.minCoeff()>=0 && seedLocations.maxCoeff()<mesh->F.rows()),
                "seedLocations contains out-of-range face indices");
        require(seedLocations.size()>0 || distRatio>=0, "distRatio must be non-negative");
        directional::streamlines_init(field, seedLocations, distRatio, data, state);
    }, "field"_a, "seedLocations"_a, "distRatio"_a, "data"_a, "state"_a, nb::keep_alive<4, 1>());

    m.def("streamlines_next", [](const StreamlineData& data, StreamlineState& state, const double dTime){
        require(data.slMesh!=nullptr, "data must be set by streamlines_init()");
        require(state.currElements.size()==data.field.N*data.sampleFaces.size(), "state does not belong to data");
        directional::streamlines_next(data, state, dTime);
    }, "data"_a, "state"_a, "dTime"_a);
}
