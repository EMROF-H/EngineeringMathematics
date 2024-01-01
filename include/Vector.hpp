#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cmath>

#include "Matrix.hpp"

namespace Math {
    template<size_t Rank>
    using RowVector = Matrix<1, Rank>;

    template<size_t Rank>
    using ColumnVector = Matrix<Rank, 1>;

    template<size_t Rank>
    using Vector = ColumnVector<Rank>;
}

#endif // VECTOR_HPP
