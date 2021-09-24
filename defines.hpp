#ifndef SECML_DEFINES_HPP
#define SECML_DEFINES_HPP
#include <Eigen/Dense>

#define SCALING_FACTOR 8192// Precision of 13 bits = 8192

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;

// Training Parameters for Linear Regression
extern int N; // Number of Training Samples
extern int N_test; // Number of Testing Samples
extern int d; // Number of Features
extern int B; // Batch Size
extern int NUM_EPOCHS; // Number of Epochs

#endif // SECML_DEFINES_HPP