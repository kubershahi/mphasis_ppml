#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>
#include <Eigen/Dense>
#include <math.h>
#include <stdlib.h>

using namespace std;
using namespace Eigen;

#define SCALING_FACTOR 16777216// Precision of 24 bits

/*

2^16 = 65536
2^20 = 1048576 
2^24 = 16777216
2^28 = 268435456

*/

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;

// function that converts a single number from double to unit64
uint64_t floattouint64(double a)
{
  uint64_t res;
  if ( a >= 0)
  {
    res = (uint64_t) (a * SCALING_FACTOR);
    // cout<< res << " is positive"<<endl;
  }
  else
  {
    a = abs(a * SCALING_FACTOR);
    res = (uint64_t) pow(2,64) - a;
    // cout<< res << " is negative"<<endl;
  }
  return res;
}

// function that converts double Matrix to unit64 Matrix
MatrixXi64 floattouint64(MatrixXd A)
{
  MatrixXi64 res(A.rows(),A.cols());

  for (int i = 0; i < A.rows(); i++)
  {
    for (int j = 0; j < A.cols(); j++)
    {
      double a = A(i,j);
      if ( a >= 0)
      {
        res(i,j) = (uint64_t) (a * SCALING_FACTOR);
        //cout<< res(i,j) << " is positive"<<endl;
      }
      else
      {
        a = abs(a * SCALING_FACTOR);
        res(i,j) = (uint64_t) pow(2,64) - a;
        //cout<< res(i,j) << " is negative"<<endl;
      }
    }
  } 
  return res;
}


//function that converts a single unit64 number to double
double uint64tofloat(uint64_t a)
{
  double res;
  if (a & (1UL << 63))
  {
    res = - ((double) pow(2,64) - a)/SCALING_FACTOR;
    //cout<< res << " is negative"<<endl;
  }
  else
  {
    res = ((double) a)/SCALING_FACTOR;
    //cout<< res << " is positive"<<endl;
  }

  return res;
}


//function that coverts unit64 matrix to double matrix
MatrixXd uint64tofloat(MatrixXi64 A)
{
  MatrixXd res(A.rows(),A.cols());
  for (int i = 0; i < A.rows(); i++)
  {
    for (int j = 0; j < A.cols(); j++)
    {
      uint64_t a = A(i,j);
      if (a & (1UL << 63))
      {
        res(i,j) = -((double) pow(2,64) - a)/SCALING_FACTOR;
        //cout<< res(i,j) << " is negative"<<endl;
      }
      else
      {
        res(i,j) = ((double) a)/SCALING_FACTOR;
        //cout<< res(i,j) << " is positive"<<endl;
      }
        
    }
  } 
  return res;
}


//function that creates shares of an integer
void share(uint64_t A, uint64_t shares[])
{
	uint64_t A_0 = rand();
	shares[0] = A_0;
	shares[1] = A - A_0;
}

//function that creates shares of integers in a matrix
void share(MatrixXi64 A, MatrixXi64 shares[])
{
	MatrixXi64 A_0 = MatrixXi64::Random(A.rows(),A.cols()) / 10000;
	shares[0] = A_0;
	shares[1] = A - A_0;
}


// For integer numbers
uint64_t rec(uint64_t A, uint64_t B)
{
	return A + B;
}


// For integer matrices
MatrixXi64 rec(MatrixXi64 A, MatrixXi64 B)
{
	return A + B;
}


uint64_t truncate(uint64_t a, int factor)
{
  uint64_t res;

  if (a & (1UL << 63))
  {
    res = (uint64_t) pow(2,64) - ( (uint64_t)pow(2,64) - a)/factor;
    //cout<< res(i,j) << " is negative"<<endl;
  }
  else
  {
    res = a/factor;
    //cout<< res(i,j) << " is positive"<<endl;
  }

  return res;
}

// function that truncates integer values in a given matrix
MatrixXi64 truncate(MatrixXi64 A, int factor)
{

  MatrixXi64 res(A.rows(),A.cols());
  for (int i = 0; i < A.rows(); i++)
  {
    for (int j = 0; j < A.cols(); j++)
    {
      uint64_t a = A(i,j);
      if (a & (1UL << 63))
      {
        res(i,j) = (uint64_t) pow(2,64) - ( (uint64_t)pow(2,64) - a)/factor;
        //cout<< res(i,j) << " is negative"<<endl;
      }
      else
      {
        res(i,j) = a/factor;
        //cout<< res(i,j) << " is positive"<<endl;
      }
        
    }
  } 
  return res;
}


// ==================================

int main(){

  double X = -10.15723545348;
  double Y = 5.23423452345;

  cout << fixed;
  cout << endl << "X: " << X << endl;
  cout << "Y: " << Y << endl;

  double Z = X * Y;
  cout << endl << "Z (X * Y: floating arithmetic): " << Z << endl << endl;

  uint64_t X_i = floattouint64(X); // mapping X to integer
  uint64_t Y_i = floattouint64(Y); // mapping Y to integer

  // no secret sharing setting
  cout << "=== No Secret Sharing Setting (mapping, mulitplying, truncating, and reverse mapping) ==="<<endl << endl;

  uint64_t Z_i = X_i * Y_i;      // multiplying X and Y
  uint64_t Z_t = truncate(Z_i, SCALING_FACTOR); // truncating Z
  // cout << "Z truncated (unshared setting): " << Z_t << endl;
  
  double Z_f = uint64tofloat(Z_t);                 // mapping Z back to double

  cout << "Z (unshared setting): " << Z_f << endl << endl;

  // secret sharing setting
  cout << "=== Secret Sharing Setting (creating shares of X and Y, multiplying and then truncating them, then recreating Z) ==="<<endl << endl;

  uint64_t shares[2];
  share(X_i, shares);               // creating shares of XX
  uint64_t X_i0 = shares[0];
  uint64_t X_i1 = shares[1];

  share(Y_i, shares);               // creating shares of XX
  uint64_t Y_i0 = shares[0];
  uint64_t Y_i1 = shares[1];

  uint64_t Z_i0 = X_i0 * Y_i0 + X_i0 * Y_i1;  // 0th share of Z
  uint64_t Z_i1 = X_i1 * Y_i0 + X_i1 * Y_i1;  // 1st share of Z

  // truncating both the shares, and then recreating
  uint64_t Z_i0_t = truncate(Z_i0, SCALING_FACTOR);
  // cout << "Truncated 0th share of Z: " << Z_i0_t << endl;

  uint64_t Z_i1_t = truncate(Z_i1, SCALING_FACTOR);
  // cout << "Truncated 1st share of Z: " << Z_i1_t << endl;


  uint64_t Z_sha_r = rec(Z_i0_t, Z_i1_t);
  // cout << endl << "Z truncated (shared setting): " << Z_sha_r << endl;

  double Z_sha_f = uint64tofloat(Z_sha_r);
  cout << "Z (shared setting): " << Z_sha_f << endl;

  return 0;
}