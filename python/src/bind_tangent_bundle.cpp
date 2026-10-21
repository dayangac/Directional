// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <directional/TangentBundle.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/IntrinsicVertexTangentBundle.h>
#include <directional/ExtrinsicVertexTangentBundle.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::TangentBundle;

//Bundles keep a pointer to their mesh, so the mesh must outlive them (keep_alive).
template <typename Bundle>
void bind_bundle(nb::module_& m, const char* name)
{
    nb::class_<Bundle, TangentBundle>(m, name)
    .def("__init__", [](Bundle* tb, const directional::TriMesh& mesh){
        new (tb) Bundle();
        tb->init(mesh);
    }, "mesh"_a, nb::keep_alive<1, 2>());
}

void bind_tangent_bundle(nb::module_& m)
{
    nb::class_<TangentBundle>(m, "TangentBundle")
    .def_ro("numSpaces", &TangentBundle::numSpaces)
    .def_ro("avgAdjLength", &TangentBundle::avgAdjLength)
    .def_prop_ro("sources", view(&TangentBundle::sources), nb::rv_policy::reference_internal)
    .def_prop_ro("normals", view(&TangentBundle::normals), nb::rv_policy::reference_internal)
    .def_prop_ro("cycleSources", view(&TangentBundle::cycleSources), nb::rv_policy::reference_internal)
    .def_prop_ro("cycleCurvatures", view(&TangentBundle::cycleCurvatures), nb::rv_policy::reference_internal)
    .def_prop_ro("local2Cycle", view(&TangentBundle::local2Cycle), nb::rv_policy::reference_internal)
    .def_prop_ro("cycles", [](const TangentBundle& tb){ return tb.cycles; });  //a scipy.sparse copy

    bind_bundle<directional::PCFaceTangentBundle>(m, "PCFaceTangentBundle");
    bind_bundle<directional::IntrinsicVertexTangentBundle>(m, "IntrinsicVertexTangentBundle");
    bind_bundle<directional::ExtrinsicVertexTangentBundle>(m, "ExtrinsicVertexTangentBundle");
}
