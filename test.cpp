#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;

int main()
{
  // // selecting certain columns of a Matrix
  // Map<MatrixXf> X1(X.data()+5,5,X.cols());

  // casting a matrix;
  // MatrixXi x = X.cast<int>();

  uint64_t g = pow(2,64);
  cout << g << endl;

  uint64_t temp = 3396261;
  uint64_t sub = g - temp;
  cout << fixed <<  sub << endl;

  uint64_t res = g - sub;
  cout << res << endl;
  
  return 0;
}