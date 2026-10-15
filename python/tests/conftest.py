# This file is part of Directional, a library for directional field processing.
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

DIRECTIONAL_ROOT = Path(__file__).parents[2]
REFERENCE_PATH = Path(__file__).parent / "reference"
CXX = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")


class Reference:
    def __init__(self, outDir):
        self.outDir = outDir

    def __call__(self, matrix):
        return np.loadtxt(self.outDir / (matrix + ".txt"), ndmin=2)


#Compiles a reference program in tests/reference against the C++ library (cached by the sources it depends on),
#runs it on the tutorial data, and returns a Reference that loads the matrices it wrote.
@pytest.fixture(scope="session")
def cpp_reference(request, tmp_path_factory):
    if CXX is None:
        pytest.skip("no C++ compiler to build the reference programs")
    outputs = {}

    def run(name):
        if name not in outputs:
            source = REFERENCE_PATH / (name + ".cpp")
            sources = [source, REFERENCE_PATH / "write.h"]+sorted((DIRECTIONAL_ROOT / "include").rglob("*.[hc]*"))
            key = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:16]
            cacheDir = (request.config.cache.mkdir("cpp_reference") if request.config.cache else tmp_path_factory.mktemp("bin"))
            binary = cacheDir / (name + "-" + key)
            if not binary.exists():
                subprocess.run([CXX, "-std=c++20", "-O2", "-DNDEBUG", "-I", str(DIRECTIONAL_ROOT / "include"),
                                "-I", str(DIRECTIONAL_ROOT / "external" / "eigen"), str(source), "-o", str(binary)], check=True)
            outDir = tmp_path_factory.mktemp(name)
            subprocess.run([str(binary), str(DIRECTIONAL_ROOT / "tutorial" / "data"), str(outDir)], check=True, stdout=subprocess.DEVNULL)
            outputs[name] = Reference(outDir)
        return outputs[name]

    return run
