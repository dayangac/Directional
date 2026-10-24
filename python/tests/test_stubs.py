# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from importlib import resources


def test_stubs_installed():
    package = resources.files("directional")
    assert package.joinpath("py.typed").is_file()
    stub = package.joinpath("_directional.pyi").read_text()
    assert "def power_field(" in stub
    assert "class CartesianField" in stub
