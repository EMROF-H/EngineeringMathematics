#ifndef TRANSPOSE_OPERATOR_HPP
#define TRANSPOSE_OPERATOR_HPP

namespace Math {
    class TransposeOperator {
    private:
        TransposeOperator() = default;
    public:
        static TransposeOperator& getInstance() {
            static TransposeOperator instance;
            return instance;
        }
    };

    const TransposeOperator T = TransposeOperator::getInstance();
}

#endif // TRANSPOSE_OPERATOR_HPP
