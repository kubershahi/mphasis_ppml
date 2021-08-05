#ifndef SECML_UTIL_HPP
#define SECML_UTIL_HPP

#include "defines.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void share(MatrixXi A, MatrixXi shares[]);
void share(MatrixXi64 A, MatrixXi64 shares[]);
void share(MatrixXd A, MatrixXd shares[]);

MatrixXi rec(MatrixXi A, MatrixXi B);
MatrixXi64 rec(MatrixXi64 A, MatrixXi64 B);
MatrixXd rec(MatrixXd A, MatrixXd B);

MatrixXi64 floattouint64(MatrixXd A);
MatrixXd uint64tofloat(MatrixXi64 A);
MatrixXi64 truncate(MatrixXi64 A, int factor);

MatrixXi mult(int i, MatrixXi A, MatrixXi B, MatrixXi E, MatrixXi F, MatrixXi Z);
MatrixXi64 mult(int i, MatrixXi64 A, MatrixXi64 B, MatrixXi64 E, MatrixXi64 F, MatrixXi64 Z);
MatrixXd mult(int i, MatrixXd A, MatrixXd B, MatrixXd E, MatrixXd F, MatrixXd Z);

MatrixXd predict(MatrixXd X, MatrixXd Y, MatrixXd w);
MatrixXd predict(MatrixXd X, MatrixXd w);
float TestAcc(int s, MatrixXd w, MatrixXd X_test, MatrixXd Y_test);

#endif // SECML_UTIL_HPP