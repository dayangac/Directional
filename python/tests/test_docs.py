# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

import re
from pathlib import Path

import pytest

DIRECTIONAL_ROOT = Path(__file__).parents[2]


#Runs the Python code blocks of docs/python.md in order, as one script from the main folder (with polyscope's mock backend).
def test_python_docs_examples_run(monkeypatch, capsys):
    ps = pytest.importorskip("polyscope")
    monkeypatch.chdir(DIRECTIONAL_ROOT)
    init = ps.init
    monkeypatch.setattr(ps, "init", lambda *args, **kwargs: init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *args, **kwargs: None)
    blocks = re.findall(r"```python\n(.*?)```", (DIRECTIONAL_ROOT / "docs" / "python.md").read_text(), re.DOTALL)
    assert len(blocks)==3
    namespace = {}
    for block in blocks:
        exec(block, namespace)
    assert "singularities with indices" in capsys.readouterr().out
    assert namespace["NFunction"].shape==(len(namespace["meshCut"].V), namespace["N"])
