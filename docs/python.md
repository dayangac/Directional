# Python bindings

Directional can be used from Python. The bindings are in the `python` folder, and are generated with [nanobind](https://nanobind.readthedocs.io/) from the same header files as the C++ library, so the results are the same as in C++. The tutorial examples have Python versions in `python/examples`, which follow the corresponding C++ tutorials (all but 701, which uses the phased-out subdivision fields).

## Installation

From the main Directional folder:

```bash
pip install ./python
```

This builds the bindings with your C++ compiler (C++20 is required), and installs `numpy` and `scipy`. The viewer helpers and the examples use [polyscope](https://polyscope.run/py/), which is installed with `pip install polyscope` (or `pip install "./python[viewer]"`). The package includes type stubs, so `help()` and editors show the signatures and default values.

Meshing (tutorial 505) uses exact arithmetic in [GMP](https://gmplib.org/), so it is only built when GMP is found, for instance from `brew install gmp` or `apt install libgmp-dev`. A different GMP installation can be given with the `GMP_DIR` environment variable, and meshing can be turned off with:

```bash
pip install ./python --config-settings=cmake.define.DIRECTIONAL_MESHING=OFF
```

## Conventions

The Python functions and classes have the same names, arguments and default values as in C++, so the documentation in the header files applies to them as well. The differences are:

1. Output arguments are return values. For instance, `directional::power_field(tb, constSpaces, constVectors, alignWeights, N, field)` is `field = power_field(tb, constSpaces, constVectors, alignWeights, N)`, and functions with several outputs return a tuple, in the order of the C++ arguments.
2. Eigen matrices are NumPy arrays, and sparse matrices are `scipy.sparse` matrices. The arrays of a mesh, a tangent bundle or a field are read-only views; they are changed through functions such as `set_extrinsic_field()`.
3. Invalid input raises `ValueError` (where C++ only asserts), and failing to read a file or to solve raises `RuntimeError`.
4. Objects that refer to each other (a tangent bundle to its mesh, a field to its tangent bundle) keep each other alive, so there is no need to keep Python references to them.
5. Arguments that have only one implemented value in C++ (such as the polynomial degree of the FEM matrices) are left out.

## Example

Computing a power field with a single constraint on a mesh, and its singularities:

```python
import numpy as np
import directional as dr

mesh = dr.readOFF("tutorial/data/bumpy.off")
ftb = dr.PCFaceTangentBundle(mesh)
N = 4
powerField = dr.power_field(ftb, np.array([0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), N)
rawField = dr.power_to_raw(powerField, N, normalize=True)
dr.principal_matching(rawField)
print(len(rawField.singLocalCycles), "singularities with indices", rawField.singIndices/N)
```

Integrating the field into a seamless parameterization:

```python
intData = dr.IntegrationData(N)
meshCut, combedField = dr.setup_integration(rawField, intData)
intData.integralSeamless = True
NFunction, NCornerFunctions = dr.integrate(combedField, intData, meshCut)
```

## Visualization

`directional.viewer` has the counterparts of the `DirectionalViewer` functions used by the tutorials, with the same colors and sizes, on top of the polyscope Python package. For instance:

```python
import polyscope as ps
import directional.viewer as viewer

ps.init()
viewer.set_surface_mesh(mesh)
viewer.set_cartesian_field(rawField)
viewer.set_isolines(meshCut, NFunction, combedField)
ps.show()
```

Instead of a field number, the functions take the polyscope names of the structures (for instance, `set_uv(UV, "UV", meshName="Cut Mesh")`), and return the polyscope structures they create, so they can be changed with the polyscope API.

## How the bindings are built

The bindings use the following tools:

- [nanobind](https://nanobind.readthedocs.io/) generates the extension module `directional._directional`, and converts Eigen matrices to NumPy arrays and Eigen sparse matrices to `scipy.sparse` matrices. It also generates the type stubs (`_directional.pyi`) at build time.
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) lets `pip` build the module with CMake (`python/pyproject.toml` and `python/CMakeLists.txt`). The compiled module includes the library headers and the bundled Eigen directly, so the C++ library does not need to be installed.
- [GMP](https://gmplib.org/) is optional, and only needed for meshing (see above).
- [cibuildwheel](https://cibuildwheel.pypa.io/) builds the binary wheels for Linux, macOS and Windows in `.github/workflows/wheels.yml`. The Linux and macOS wheels bundle GMP (with auditwheel and delocate), and the Windows wheels are built without meshing.
- [polyscope](https://polyscope.run/py/) is used by `directional.viewer`, which is pure Python.
- [pytest](https://pytest.org/) runs the tests.

The code is organized as follows:

```
python/
  pyproject.toml      package and wheel configuration
  CMakeLists.txt      module, optional meshing, type stubs
  src/module.cpp      creates the module and calls the bind_* functions
  src/bind_*.cpp      one file per part of the library (mesh, tangent bundles, fields, streamlines,
                      matching, PolyVectors, FEM, DEC, integration, meshing)
  directional/        __init__.py (re-exports the module) and viewer.py
  examples/           Python versions of the tutorial examples
  tests/              pytest tests, and C++ reference programs in tests/reference
```

The bindings are kept thin. Every class and function is bound directly from the headers, and a short wrapper is written only to return output arguments, to keep referenced objects alive, or to check input that C++ only asserts. Internal helpers, the deprecated headers and the C++ viewer are not bound.

When a binding is added or changed in C++, the module is rebuilt with `pip install ./python`. For faster rebuilds during development:

```bash
pip install nanobind scikit-build-core numpy scipy
pip install --no-build-isolation ./python
```

## Tests

The tests are in `python/tests` and run with:

```bash
pip install pytest polyscope
pytest python/tests
```

Besides testing properties of the results, the tests compile the small C++ programs in `python/tests/reference` against the library, and compare their output to the Python results (they are skipped when there is no C++ compiler).
