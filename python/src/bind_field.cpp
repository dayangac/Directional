#include <fstream>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <directional/CartesianField.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>
#include <directional/principal_matching.h>
#include <directional/read_raw_field.h>
#include <directional/write_raw_field.h>
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using directional::CartesianField;
using directional::TangentBundle;
using directional::fieldTypeEnum;

// Number of vectors stored per tangent space (a power field stores the single complex u^N).
static int vectors_per_space(const CartesianField& field) {
    return field.fieldType == fieldTypeEnum::POWER_FIELD ? 1 : field.N;
}

// Fields keep a pointer to their tangent bundle; keep_alive ties their lifetimes together.
void bind_field(nb::module_& m) {
    nb::enum_<fieldTypeEnum>(m, "fieldTypeEnum")
        .value("RAW_FIELD", fieldTypeEnum::RAW_FIELD)
        .value("POWER_FIELD", fieldTypeEnum::POWER_FIELD)
        .value("POLYVECTOR_FIELD", fieldTypeEnum::POLYVECTOR_FIELD);

    nb::class_<CartesianField>(m, "CartesianField")
        .def("__init__", [](CartesianField* field, const TangentBundle& tb, fieldTypeEnum fieldType, int N) {
            require(N > 0, "N must be positive");
            new (field) CartesianField();
            field->init(tb, fieldType, N);
        }, "tb"_a, "fieldType"_a, "N"_a, nb::keep_alive<1, 2>())
        .def_ro("N", &CartesianField::N)
        .def_ro("fieldType", &CartesianField::fieldType)
        .def_prop_ro("intField", view(&CartesianField::intField), nb::rv_policy::reference_internal)
        .def_prop_ro("extField", view(&CartesianField::extField), nb::rv_policy::reference_internal)
        .def_prop_ro("matching", view(&CartesianField::matching), nb::rv_policy::reference_internal)
        .def_prop_ro("effort", view(&CartesianField::effort), nb::rv_policy::reference_internal)
        .def_prop_ro("singLocalCycles", view(&CartesianField::singLocalCycles), nb::rv_policy::reference_internal)
        .def_prop_ro("singIndices", view(&CartesianField::singIndices), nb::rv_policy::reference_internal)
        .def("set_intrinsic_field", [](CartesianField& field, const Eigen::MatrixXd& intField) {
            require(intField.rows() == field.tb->numSpaces && intField.cols() == 2 * vectors_per_space(field),
                    "intField must be #spaces x 2N (#spaces x 2 for power fields)");
            field.set_intrinsic_field(intField);
        }, "intField"_a)
        .def("set_extrinsic_field", [](CartesianField& field, const Eigen::MatrixXd& extField) {
            require(extField.rows() == field.tb->numSpaces && extField.cols() == 3 * vectors_per_space(field),
                    "extField must be #spaces x 3N (#spaces x 3 for power fields)");
            field.set_extrinsic_field(extField);
        }, "extField"_a);

    m.def("power_field", [](const TangentBundle& tb, const Eigen::VectorXi& constSpaces,
                            const Eigen::MatrixXd& constVectors, const Eigen::VectorXd& alignWeights,
                            int N, bool normalizeField) {
        require(N > 0, "N must be positive");
        require(constVectors.cols() == 3, "constVectors must be #constSpaces x 3");
        require(constVectors.rows() == constSpaces.size() && alignWeights.size() == constSpaces.size(),
                "constSpaces, constVectors and alignWeights must have the same length");
        require(constSpaces.size() == 0 || (constSpaces.minCoeff() >= 0 && constSpaces.maxCoeff() < tb.numSpaces),
                "constSpaces contains out-of-range indices");
        CartesianField field;
        directional::power_field(tb, constSpaces, constVectors, alignWeights, N, field, normalizeField);
        return field;
    }, "tb"_a, "constSpaces"_a, "constVectors"_a, "alignWeights"_a, "N"_a, "normalizeField"_a = false,
       nb::keep_alive<0, 1>());

    m.def("power_to_raw", [](const CartesianField& powerField, int N, bool normalize) {
        require(powerField.fieldType == fieldTypeEnum::POWER_FIELD, "powerField must be a POWER_FIELD");
        require(N > 0, "N must be positive");
        CartesianField rawField;
        directional::power_to_raw(powerField, N, rawField, normalize);
        return rawField;
    }, "powerField"_a, "N"_a, "normalize"_a = false, nb::keep_alive<0, 1>());

    m.def("principal_matching", [](CartesianField& field, bool isSingularities) {
        require(field.fieldType == fieldTypeEnum::RAW_FIELD, "field must be a RAW_FIELD");
        directional::principal_matching(field, isSingularities);
    }, "field"_a, "isSingularities"_a = true);

    m.def("read_raw_field", [](const std::string& fileName, const TangentBundle& tb) {
        int N, numSpaces;
        std::ifstream header(fileName);
        if (!(header >> N >> numSpaces))
            throw std::runtime_error("could not read " + fileName);
        require(numSpaces == tb.numSpaces, "the file's field size does not match the tangent bundle");
        CartesianField field;
        if (!directional::read_raw_field(fileName, tb, N, field))
            throw std::runtime_error("could not read " + fileName);
        return field;
    }, "fileName"_a, "tb"_a, nb::keep_alive<0, 2>());

    m.def("write_raw_field", [](const std::string& fileName, const CartesianField& rawField, bool high_precision) {
        if (!directional::write_raw_field(fileName, rawField, high_precision))
            throw std::runtime_error("could not write " + fileName);
    }, "fileName"_a, "rawField"_a, "high_precision"_a = false);
}
