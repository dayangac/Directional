cmake_minimum_required(VERSION 3.20)
project(directional_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

set(DIRECTIONAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

nanobind_add_module(_directional
    src/module.cpp
    src/bind_mesh.cpp
    src/bind_tangent_bundle.cpp
    src/bind_field.cpp
    src/bind_streamlines.cpp
    src/bind_matching.cpp
    src/bind_polyvector.cpp
    src/bind_fem.cpp
    src/bind_dec.cpp
    src/bind_integration.cpp
)
target_include_directories(_directional PRIVATE
    "${DIRECTIONAL_ROOT}/include"
    "${DIRECTIONAL_ROOT}/external/eigen"
)

install(TARGETS _directional LIBRARY DESTINATION directional)
