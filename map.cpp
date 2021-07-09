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
    res = (a * SCALING_FACTOR);
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
    res = -((double) pow(2,64) - a)/SCALING_FACTOR;
    //cout<< res(i,j) << " is negative"<<endl;
  }
  else
  {
    res = ((double) a)/SCALING_FACTOR;
    //cout<< res(i,j) << " is positive"<<endl;
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


// function that truncates integer values
uint64_t truncate(uint64_t a, int factor)
{ 
  cout<< "G: "<<G<<endl;
  uint64_t res;
  if (a & (1UL << 63))
  {
    res = (uint64_t) pow(2,64) - ((uint64_t) pow(2,64) - a)/factor;
    //cout << res << " is negative"<<endl;
  }
  else
  {
    res = a/factor;
    //cout<< res << " is positive"<<endl;
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


// function that truncates the shares
uint64_t truncate_share(uint64_t a, int factor, int i)
{
  uint64_t res;
  if (i == 0){
    res = truncate(a, SCALING_FACTOR); // truncation for 0th share
  }
  else{ 
    res = (uint64_t) pow(2,64) - truncate( (uint64_t) pow(2,64) - a, SCALING_FACTOR); // truncation for 1st share
  }

  return res;
}

// ==================================

int main(){

  uint64_t G = (uint64_t) pow(2,64) - 2;
  cout<< "G (in main): "<<G<<endl;

  double XX = 10.15723545348;
  double YY = -5.23423452345;
  double ZZ = XX * YY;

  cout << fixed;
  cout << "Z (floating arithmetic): " << ZZ << endl << endl;

  uint64_t XX_i = floattouint64(XX); // mapping XX to integer
  uint64_t YY_i = floattouint64(YY); // mapping YY to integer

  // no secret sharing setting
  uint64_t ZZ_i = XX_i * YY_i;      // multiplying X and Y
  uint64_t ZZ_t = truncate(ZZ_i, SCALING_FACTOR); // truncating Z
  cout << "Z truncated (unshared setting)" << ZZ_t << endl;
  double ZZ_f = uint64tofloat(ZZ_t);                 // mapping Z back to double

  cout << "Z (unshared setting): " << ZZ_f << endl;

  // secret sharing setting
  uint64_t shares[2];
  share(XX_i, shares);               // creating shares of XX
  uint64_t XX_i0 = shares[0];
  uint64_t XX_i1 = shares[1];

  share(YY_i, shares);               // creating shares of XX
  uint64_t YY_i0 = shares[0];
  uint64_t YY_i1 = shares[1];

  uint64_t ZZ_i0 = XX_i0 * YY_i0 + XX_i0 * YY_i1;  // 0th share of Z
  uint64_t ZZ_i1 = XX_i1 * YY_i0 + XX_i1 * YY_i1;  // 1st share of Z

  // // without truncation of shares, recreating Z first and then truncation
  // uint64_t ZZ_rec = rec(ZZ_i0, ZZ_i1);
  // uint64_t ZZ_trun = truncate(ZZ_rec, SCALING_FACTOR);
  // cout << endl << "Z truncated (shared setting) " << ZZ_trun << endl;
  // double ZZ_sf = uint64tofloat(ZZ_trun);

  // cout << "Z (shares recreated, then truncated): " << ZZ_sf << endl;

  // truncating both the shares, and then recreating
  uint64_t ZZ_i0_t = truncate_share(ZZ_i0, SCALING_FACTOR, 0);
  cout << endl << "Truncated 0th share of Z: " << ZZ_i0_t << endl;
  uint64_t ZZ_i1_t = truncate_share(ZZ_i1, SCALING_FACTOR, 1);
  cout << "Truncated 1st share of Z (treating as negative representation): " << ZZ_i1_t << endl;

  // //truncating first share manually according to paper
  // uint64_t ZZ_i1_sub = (uint64_t) ((uint64_t) pow(2,64) - ZZ_i1)/SCALING_FACTOR;
  // uint64_t ZZ_i1_trun_test = (uint64_t) pow(2,64) - ZZ_i1_sub;
  
  // cout << "1st share of Z (treating as postive number): " << ZZ_i1_trun_test << endl;

  // uint64_t sum = (ZZ_i0_t + ZZ_i1_trun_test) ;
  // cout << "Sum: " << sum << endl;

  // uint64_t result = (uint64_t) pow(2,64) - sum;
  // cout << "Result: " << result << endl;

  uint64_t ZZ_r = rec(ZZ_i0_t, ZZ_i1_t);
  cout << endl << "Z truncated (shared setting) " << ZZ_r << endl;

  double ZZ_shaf = uint64tofloat(ZZ_r);
  cout << "Z (shares truncated, then recreated): " << ZZ_shaf << endl;

  return 0;
}