#ifndef SECML_LINEAR_REGRESSION_HPP
#define SECML_LINEAR_REGRESSION_HPP

#include "read_data.hpp"
#include "defines.hpp"
#include "utils.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXi idealLinearRegression(MatrixXi X, MatrixXi Y, MatrixXi w);
MatrixXd idealLinearRegression(MatrixXd X, MatrixXd Y, MatrixXd w);
MatrixXi64 idealLinearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w);

MatrixXi64 linearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w);
MatrixXd linearRegression(MatrixXd X, MatrixXd Y, MatrixXd w); 

#endif // SECML_UTIL_HPP