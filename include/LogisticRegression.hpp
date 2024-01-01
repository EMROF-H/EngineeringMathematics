#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <cmath>

#include "Matrix.hpp"
#include "Vector.hpp"

namespace Math {
    template<size_t Dimension>
    class LogisticRegression {
    private:
        Vector<Dimension> weight {};
        double LearningRate;

    public:
        inline double possibilityPositive(const Vector<Dimension> &x) {
            return 1 / (1 + std::exp((weight ^ T) * x));
        }

    private:
        template<size_t DataNumber>
        Vector<DataNumber> possibilityPositive(const Matrix<DataNumber, Dimension> &X) {
            Vector<DataNumber> result;
            for (int i = 0; i < DataNumber; i++) {
                auto x = X.SubRow(i) ^ T;
                result[i] = possibilityPositive(x);
            }
            return result;
        }

        template<size_t DataNumber>
        inline Vector<Dimension> gradient(
                const Matrix<DataNumber, Dimension> &X,
                const Vector<DataNumber> &y,
                const Vector<DataNumber> &p) {
            return (X ^ T) * (y - p);
        }

        template<size_t DataNumber>
        inline void fitOnce(
                const Matrix<DataNumber, Dimension> &inputData,
                const Vector<DataNumber> &outputData) {
            auto p = possibilityPositive(inputData);
            auto g = gradient(inputData, outputData, p);
            weight -= g * LearningRate;
        }

    public:
        explicit LogisticRegression(double learningRate) : LearningRate(learningRate) { }

        explicit LogisticRegression(double learningRate, double initWeight) :
            LearningRate(learningRate) {
            for (int i = 0; i < Dimension; ++i) {
                weight[i] = initWeight;
            }
        }

        explicit LogisticRegression(double learningRate,
                                    std::initializer_list<double> initWeights) :
            LearningRate(learningRate) {

            if (initWeights.size() != Dimension) {
                throw std::invalid_argument("Invalid dimensions for initialization");
            }

            size_t index = 0;
            for (const auto& v : initWeights) {
                this->weight[index] = v;
                index++;
            }
        }

        [[nodiscard]] double getLearningRate() const { return LearningRate; }

        void setLearningRate(double learningRate) { LearningRate = learningRate; }

        [[nodiscard]] const Vector<Dimension> &getWeight() const { return weight; }

        template<size_t DataNumber>
        void fit(const Matrix<DataNumber, Dimension + 1> &data, size_t fitTimes) {
            const auto X = data.template SubColumns<Dimension>(0);
            const auto r = data.SubColumn(Dimension);

            for (int i = 0; i < fitTimes; ++i) {
                fitOnce(X, r);
            }
        }
    };
}

#endif // LOGISTIC_REGRESSION_HPP
