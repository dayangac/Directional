# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

import runpy
from pathlib import Path

import pytest

ps = pytest.importorskip("polyscope")

EXAMPLES_PATH = Path(__file__).parents[1] / "examples"


def run_example(name, monkeypatch):
    init = ps.init
    monkeypatch.setattr(ps, "init", lambda *args: None if ps.is_initialized() else init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *args: None)
    example = runpy.run_path(str(EXAMPLES_PATH / name))
    return example["update_visualization"].__globals__


def test_305_shows_only_the_singularities_of_the_viewed_field(monkeypatch):
    example = run_example("305_extrinsic_fields.py", monkeypatch)
    for mode, expected in ((example["FACE_FIELD"], {"Face-Based Field singularities"}),
                           (example["VERTEX_FIELD"], {"Exact Singularities"}),
                           (example["INTERPOLATED_FIELD"], set())):
        example["viewingMode"] = mode
        example["update_visualization"]()
        shown = {name for name in ("Face-Based Field singularities", "Vertex-Based Field singularities", "Exact Singularities")
                 if ps.has_point_cloud(name) and ps.get_point_cloud(name).is_enabled()}
        assert shown==expected
    ps.remove_all_structures()
