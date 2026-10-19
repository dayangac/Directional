// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_mesh(nb::module_& m);
void bind_tangent_bundle(nb::module_& m);
void bind_field(nb::module_& m);
void bind_streamlines(nb::module_& m);
void bind_matching(nb::module_& m);
void bind_polyvector(nb::module_& m);

NB_MODULE(_directional, m)
{
    bind_mesh(m);
    bind_tangent_bundle(m);
    bind_field(m);
    bind_streamlines(m);
    bind_matching(m);
    bind_polyvector(m);
}
