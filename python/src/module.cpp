#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_mesh(nb::module_& m);
void bind_tangent_bundle(nb::module_& m);
void bind_field(nb::module_& m);

NB_MODULE(_directional, m) {
    bind_mesh(m);
    bind_tangent_bundle(m);
    bind_field(m);
}
