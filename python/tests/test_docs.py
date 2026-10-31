# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

import re
from pathlib import Path

import pytest

DIRECTIONAL_ROOT = Path(__file__).parents[2]


#The Python tabs of docs/tutorial.md show lines of python/examples, so each of their code lines has to be found, in the
#same order, in one example. Lines like "pvData.wSmooth" list the fields of a data structure, which have to exist.
def test_tutorial_python_tabs_match_the_examples():
    dr = pytest.importorskip("directional")
    dataStructures = {"pvData": dr.PolyVectorData, "intData": dr.IntegrationData}
    examples = [path.read_text().splitlines() for path in sorted((DIRECTIONAL_ROOT / "python" / "examples").glob("*.py"))]
    tabs = re.findall(r'=== "Python"\n\n((?:    .*\n|\n)*)', (DIRECTIONAL_ROOT / "docs" / "tutorial.md").read_text())
    blocks = [re.search(r"```python\n(.*?)```", tab, re.DOTALL).group(1) for tab in tabs if "```python" in tab]
    assert len(blocks)==13
    for block in blocks:
        lines = [line.strip() for line in block.splitlines()]
        lines = [line for line in lines if line and line!="..." and not line.startswith("#")]
        fields = [re.fullmatch(r"(\w+)\.(\w+)\s*(#.*)?", line) for line in lines]
        if all(fields):
            for field in fields:
                assert hasattr(dataStructures[field.group(1)], field.group(2)), field.group(0)
            continue
        assert any(is_subsequence(lines, [line.strip() for line in example]) for example in examples), block


def is_subsequence(lines, exampleLines):
    remaining = iter(exampleLines)
    return all(line in remaining for line in lines)
