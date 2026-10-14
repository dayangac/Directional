#pragma once

#include <stdexcept>
#include <Eigen/Core>

// Raises ValueError in Python; used where the C++ library only asserts.
inline void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

// Getter exposing an Eigen member as a read-only NumPy view (use with rv_policy::reference_internal).
// Writable views would let Python change data that other C++ quantities were computed from.
template <typename Class, typename Matrix>
auto view(Matrix Class::*member) {
    return [member](const Class& self) {
        const Matrix& a = self.*member;
        return Eigen::Map<const Matrix>(a.data(), a.rows(), a.cols());
    };
}
