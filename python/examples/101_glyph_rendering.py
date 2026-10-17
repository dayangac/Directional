# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Python port of tutorial/101_GlyphRendering: reads a field and its singularities, and shows them as glyphs.
from pathlib import Path

import polyscope as ps

import directional as dr
import directional.viewer as viewer

TUTORIAL_DATA_PATH = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOFF(str(TUTORIAL_DATA_PATH / "bumpy.off"))
ftb = dr.PCFaceTangentBundle(mesh)
field = dr.read_raw_field(str(TUTORIAL_DATA_PATH / "bumpy.rawfield"), ftb)
dr.read_singularities(str(TUTORIAL_DATA_PATH / "bumpy.sings"), field)

ps.init()
viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(field)
ps.show()
