#ifndef SECML_LOGISTIC_REGRESSION_HPP
#define SECML_LOGISTIC_REGRESSION_HPP

#include "defines.hpp"
#include "utils.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd idealLogisticRegression(MatrixXd X, MatrixXd Y, MatrixXd w);
MatrixXi64 idealLinearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w);

MatrixXd linearRegression(MatrixXd X, MatrixXd Y, MatrixXd w); 
MatrixXi64 linearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w);

#endif // SECML_UTIL_HPP