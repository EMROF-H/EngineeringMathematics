#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <array>
#include <valarray>

#include "TransposeOperator.hpp"

namespace Math {
    template<size_t Rows, size_t Cols>
    class Matrix {
    private:
        std::array<std::array<double, Cols>, Rows> value {};

    public:
        #pragma region Matrix
        #pragma region Constructor
        Matrix() = default;

        Matrix(std::initializer_list<std::initializer_list<double>> initValues) {
            if (initValues.size() != Rows || initValues.begin()->size() != Cols) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }

            size_t rowIdx = 0;
            for (const auto& row : initValues) {
                if (row.size() != Cols) {
                    throw std::invalid_argument("Invalid number of columns");
                }

                size_t colIdx = 0;
                for (const auto& v : row) {
                    value[rowIdx][colIdx] = v;
                    colIdx++;
                }
                rowIdx++;
            }
        }

        Matrix(std::initializer_list<Matrix<1, Cols>> initValues) requires(Rows != 1) {
            if (initValues.size() != Rows) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }

            size_t rowIdx = 0;
            for (const auto& row : initValues) {
                for (int colIdx = 0; colIdx < Cols; colIdx++) {
                    value[rowIdx][colIdx] = row[colIdx];
                }
                rowIdx++;
            }
        }

        Matrix(std::initializer_list<Matrix<Rows, 1>> initValues) requires(Cols != 1) {
            if (initValues.size() != Cols) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }

            size_t colIdx = 0;
            for (const auto& col : initValues) {
                for (int rowIdx = 0; rowIdx < Rows; rowIdx++) {
                    value[rowIdx][colIdx] = col[rowIdx];
                }
                colIdx++;
            }
        }
        #pragma endregion

        #pragma region Property
        [[nodiscard]] size_t getRows() const { return Rows; }

        [[nodiscard]] size_t getCols() const { return Cols; }
        #pragma endregion

        #pragma region Operator
        double &operator[](int r, int c) {
            return value[r][c];
        }

        const double &operator[](int r, int c) const {
            return value[r][c];
        }

        Matrix operator+(const Matrix &other) const {
            Matrix result;
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    result[i, j] = (*this)[i, j] + other.value[i][j];
                }
            }
            return result;
        }

        void operator+=(const Matrix &other) {
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    (*this)[i, j] += other[i, j];
                }
            }
        }

        Matrix operator-(const Matrix &other) const {
            Matrix result;
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    result[i, j] = (*this)[i, j] - other[i, j];
                }
            }
            return result;
        }

        void operator-=(const Matrix &other) {
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    (*this)[i, j] -= other[i, j];
                }
            }
        }

        Matrix<Cols, Rows> operator^(const TransposeOperator &) const
            requires(!(Rows == 1 && Cols == 1)) {
            Matrix<Cols, Rows> transposed;
            for (size_t i = 0; i < Cols; ++i) {
                for (size_t j = 0; j < Rows; ++j) {
                    transposed[i, j] = (*this)[j, i];
                }
            }
            return transposed;
        }

        Matrix operator*(double number) const {
            Matrix result;
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) {
                    result[i, j] = (*this)[i, j] * number;
                }
            }
            return result;
        }

        friend Matrix operator*(double number, const Matrix &matrix) {
            Matrix result;
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) {
                    result[i, j] = number * matrix[i, j];
                }
            }
            return result;
        }

        template<size_t OtherCols>
        Matrix<Rows, OtherCols> operator*(const Matrix<Cols, OtherCols> &other) const
            requires(!(Rows == 1 && Cols == 1) && !(Cols == 1 && OtherCols == 1)) {
            Matrix<Rows, OtherCols> result;
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < OtherCols; ++j) {
                    for (size_t k = 0; k < Cols; ++k) {
                        result[i, j] += (*this)[i, k] * other[k, j];
                    }
                }
            }
            return result;
        }

        friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
            for (size_t i = 0; i < Rows; ++i) {
                if (i != 0) {
                    std::cout << std::endl;
                }
                for (size_t j = 0; j < Cols; ++j) {
                    if (j != 0) {
                        std::cout << " ";
                    }
                    std::cout << matrix.value[i][j];
                }
            }
            return os;
        }
        #pragma endregion

        #pragma region Method
        Matrix<1, Cols> SubRow(size_t index) const {
            if (index >= Rows) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }
            return SubRows<1>(index);
        }

        template<size_t Row>
        Matrix<Row, Cols> SubRows(size_t index) const {
            if (index + Row - 1 >= Rows) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }
            Matrix<Row, Cols> result;
            for (int i = 0; i < Row; i++) {
                for (int j = 0; j < Cols; j++) {
                    result[i, j] = (*this)[index + i, j];
                }
            }
            return result;
        }

        Matrix<Rows, 1> SubColumn(size_t index) const {
            if (index >= Cols) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }
            return SubColumns<1>(index);
        }

        template<size_t Col>
        Matrix<Rows, Col> SubColumns(size_t index) const {
            if (index + Col - 1 >= Cols) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }
            Matrix<Rows, Col> result;
            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Col; j++) {
                    result[i, j] = (*this)[i, index + j];
                }
            }
            return result;
        }

        void fill(double v) {
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) {
                    (*this)[i, j] = v;
                }
            }
        }
        #pragma endregion
        #pragma endregion

        #pragma region Vector
        #pragma region Constructor
        Matrix(std::initializer_list<double> initValues) requires(Rows == 1) {
            if (initValues.size() != Cols) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }

            size_t colIdx = 0;
            for (const auto& v : initValues) {
                this->value[0][colIdx] = v;
                colIdx++;
            }
        }

        Matrix(std::initializer_list<double> initValues) requires(Cols == 1) {
            if (initValues.size() != Rows) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }

            size_t rowIdx = 0;
            for (const auto& v : initValues) {
                this->value[rowIdx][0] = v;
                rowIdx++;
            }
        }
        #pragma endregion

        #pragma region Property
        size_t getRank() requires(Rows == 1) { return Cols; };

        size_t getRank() requires(Cols == 1) { return Rows; };
        #pragma endregion

        #pragma region Operator
        double &operator[](size_t index) requires(Rows == 1) {
            return this->value[0][index];
        }

        const double &operator[](size_t index) const requires(Rows == 1) {
            return this->value[0][index];
        }

        double &operator[](size_t index) requires(Cols == 1) {
            return this->value[index][0];
        }

        const double &operator[](size_t index) const requires(Cols == 1) {
            return this->value[index][0];
        }
        #pragma endregion

        #pragma region Method
        double abs() requires(Rows == 1 || Cols == 1) {
            double sum = 0;
            for (int i = 0; i < getRank(); i++) {
                sum += ((*this)[i]) * ((*this)[i]);
            }
            return sqrt(sum);
        }
        #pragma endregion
        #pragma endregion

        operator double() const requires(Rows == 1 && Cols == 1) { // NOLINT(*-explicit-constructor)
            return (*this)[0, 0];
        }
    };
}

#endif // MATRIX_HPP
