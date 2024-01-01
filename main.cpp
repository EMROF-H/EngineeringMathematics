#include <iostream>

#include "Matrix.hpp"
#include "Vector.hpp"
#include "LogisticRegression.hpp"

using std::cout;
using std::endl;
using namespace Math;

const double DecisionThreshold = 0.5;   // 判决门限

const size_t DataNumber = 9;    // 原始数据数量
const size_t Dimension = 5;     // 数据维度

using RawDataType = const Matrix<DataNumber, Dimension + 1>;
RawDataType &getRawData() { // 原始样本数据
    static RawDataType RawData {
        { 4,  3.4,  100,  3,  10, 1 },
        { 6,  4.1,  210,  1,  8 , 1 },
        { 8,  6.7,  600,  2,  16, 0 },
        { 10, 8.5,  1600, 6,  11, 0 },
        { 5,  4.8,  150,  13, 12, 0 },
        { 18, 15.6, 120,  21, 20, 1 },
        { 2,  3.4,  80,   1,  10, 1 },
        { 12, 7.9,  600,  4,  11, 0 },
        { 16, 12,   780,  8,  8 , 0 },
    };
    return RawData;
}

inline std::string result(double possibility) {
    if (possibility >= DecisionThreshold) {
        return "Positive";
    } else {
        return "Negative";
    }
}

int main() {
    const Vector<Dimension> Sample10 { 9, 15,  800, 7,  16 };   // 待预测样本
    const Vector<Dimension> Sample11 { 3, 4.2, 189, 11, 7  };   // 待预测样本

    const size_t FitTimes = 100000;     // 拟合次数
    const double LearningRate = 0.1;    // 学习率

    LogisticRegression<Dimension> lr(LearningRate, 1);  // 初始化Logistic回归对象，初始权重设为全1

    // 打印原始样本数据
    cout << "RawData: " << DataNumber << " * " << Dimension << endl;
    cout << getRawData() << endl;
    cout << endl;

    // 打印拟合参数
    cout << "InitWeight = " << "[" << (lr.getWeight() ^ T) << "]" << endl;
    cout << "FitTimes = " << FitTimes << endl;
    cout << "LearningRate = " << LearningRate << endl;
    cout << "DecisionThreshold = " << DecisionThreshold << endl;
    cout << endl;

    lr.fit<DataNumber>(getRawData(), FitTimes); // 使用原始样本数据进行拟合

    auto possibility10 = lr.possibilityPositive(Sample10);  // 预测样本10
    auto possibility11 = lr.possibilityPositive(Sample11);  // 预测样本11

    // 打印结果
    cout << "ResultWeight = " << "[" << (lr.getWeight() ^ T) << "]" << endl;  // 打印训练结果权重
    cout << endl;

    cout << "Sample10 = " << "[" << (Sample10 ^ T) << "]" << endl;
    cout << "Possibility = " << possibility10 << ", " << result(possibility10) << endl;
    cout << endl;

    cout << "Sample11 = " << "[" << (Sample11 ^ T) << "]" << endl;
    cout << "Possibility = " << possibility11 << ", " << result(possibility11) << endl;
    cout << endl;

    return 0;
}
