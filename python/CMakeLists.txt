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

#Meshing uses exact arithmetic in GMP, so it is only built when GMP is found (GMP_DIR can point to it).
option(DIRECTIONAL_MESHING "Build meshing if GMP is found" ON)
if(DIRECTIONAL_MESHING)
    find_path(GMP_INCLUDE_DIR NAMES gmpxx.h HINTS $ENV{GMP_DIR}/include)
    find_library(GMP_LIBRARY NAMES gmp HINTS $ENV{GMP_DIR}/lib)
    find_library(GMPXX_LIBRARY NAMES gmpxx HINTS $ENV{GMP_DIR}/lib)
endif()
if(DIRECTIONAL_MESHING AND GMP_INCLUDE_DIR AND GMP_LIBRARY AND GMPXX_LIBRARY)
    message(STATUS "GMP found, building meshing: ${GMP_INCLUDE_DIR}")
    target_sources(_directional PRIVATE src/bind_meshing.cpp)
    target_compile_definitions(_directional PRIVATE USE_GMP_ENABLED DIRECTIONAL_MESHING)
    target_include_directories(_directional PRIVATE "${GMP_INCLUDE_DIR}")
    target_link_libraries(_directional PRIVATE "${GMPXX_LIBRARY}" "${GMP_LIBRARY}")
else()
    message(WARNING "GMP not found, building without meshing")
endif()

install(TARGETS _directional LIBRARY DESTINATION directional)
