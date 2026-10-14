#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <directional/TangentBundle.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/IntrinsicVertexTangentBundle.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::TangentBundle;

// Bundles keep a pointer to their mesh, so the mesh must outlive them (keep_alive).
template <typename Bundle>
void bind_bundle(nb::module_& m, const char* name) {
    nb::class_<Bundle, TangentBundle>(m, name)
        .def("__init__", [](Bundle* tb, const directional::TriMesh& mesh) {
            new (tb) Bundle();
            tb->init(mesh);
        }, "mesh"_a, nb::keep_alive<1, 2>());
}

void bind_tangent_bundle(nb::module_& m) {
    nb::class_<TangentBundle>(m, "TangentBundle")
        .def_ro("numSpaces", &TangentBundle::numSpaces)
        .def_prop_ro("sources", view(&TangentBundle::sources), nb::rv_policy::reference_internal)
        .def_prop_ro("normals", view(&TangentBundle::normals), nb::rv_policy::reference_internal);

    bind_bundle<directional::PCFaceTangentBundle>(m, "PCFaceTangentBundle");
    bind_bundle<directional::IntrinsicVertexTangentBundle>(m, "IntrinsicVertexTangentBundle");
}
