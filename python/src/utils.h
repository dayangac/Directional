// This file is part of Directional, a library for directional field processing.
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DIRECTIONAL_PYTHON_UTILS_H
#define DIRECTIONAL_PYTHON_UTILS_H

#include <stdexcept>
#include <Eigen/Core>

//Raises ValueError in Python. Used where the library itself only asserts.
inline void require(const bool condition, const char* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

//Getter that exposes an Eigen member as a read-only NumPy view (use with rv_policy::reference_internal).
//Writable views would let Python change data that other quantities were computed from.
template <typename Class, typename Matrix>
auto view(Matrix Class::*member)
{
    return [member](const Class& self){
        const Matrix& a = self.*member;
        return Eigen::Map<const Matrix>(a.data(), a.rows(), a.cols());
    };
}

#endif
