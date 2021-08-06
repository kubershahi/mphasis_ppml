#include "utils.hpp"

#include <iostream>
#include <string>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// ====================================
// Arithmetic Secret Sharing: 
// ==================================== 

// For integer inputs
void share(MatrixXi A, MatrixXi shares[])
{
  MatrixXi A_0 = MatrixXi::Random(A.rows(),A.cols()) / 10000000;
  shares[0] = A_0;
  shares[1] = A - A_0;
}

// For 64-integer inputs
void share(MatrixXi64 A, MatrixXi64 shares[])
{
	MatrixXi64 A_0 = MatrixXi64::Random(A.rows(),A.cols()) / 10000000;
	shares[0] = A_0;
	shares[1] = A - A_0;
}

// For floating point inputs
void share(MatrixXd A, MatrixXd shares[])
{
  MatrixXd A_0 = MatrixXd::Random(A.rows(),A.cols());
  shares[0] = A_0;
  shares[1] = A - A_0;
}

// For integer inputs
MatrixXi rec(MatrixXi A, MatrixXi B)
{
	return A + B;
}

// For 64-integer inputs
MatrixXi64 rec(MatrixXi64 A, MatrixXi64 B)
{
  return A + B;
}

// For floating point inputs
MatrixXd rec(MatrixXd A, MatrixXd B)
{
  return A + B;
}

// =====================================================
// Helper Functions for Floating Point Number Operations: 
// ===================================================== 

// function that converts double Matrix to uint64 Matrix
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

//function that coverts uint64 matrix to double matrix
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

// function that truncates integer values in a given matrix by a factor
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

// ==================================================
// Secure Multiplication using Beavers' Triplets: 
// ================================================== 

// For integer inputs
MatrixXi mult(int i, MatrixXi A, MatrixXi B, MatrixXi E, MatrixXi F, MatrixXi Z)
{ 
  MatrixXi product = MatrixXi::Random(Z.rows(),Z.cols());
  if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
  else if (i == 0) product = (A * F) + (E * B) + Z;
  
  return product;
}

// For 64-integer inputs
MatrixXi64 mult(int i, MatrixXi64 A, MatrixXi64 B, MatrixXi64 E, MatrixXi64 F, MatrixXi64 Z)
{ 
	MatrixXi64 product = MatrixXi64::Random(Z.rows(),Z.cols());
	if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
	else if (i == 0) product = (A * F) + (E * B) + Z;
	
	return product;
}

// For floating point numbersinputs
MatrixXd mult(int i, MatrixXd A, MatrixXd B, MatrixXd E, MatrixXd F, MatrixXd Z)
{ 
  MatrixXd product = MatrixXd::Random(Z.rows(),Z.cols());
  if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
  else if (i == 0) product = (A * F) + (E * B) + Z;
  
  return product;
}

// ==================================================
// Helper Functions for Linear Regression: 
// ================================================== 

// function that predicts the output given trained weights and computes the loss
MatrixXd predict(MatrixXd X, MatrixXd Y, MatrixXd w)
{ MatrixXd pred = X * w;
  MatrixXd diff = pred - Y;
  MatrixXd loss = diff.transpose() * diff;
  cout<< "Test Loss: "<< loss(0,0)/X.rows() << endl;
  return pred;
}

//function that just predicts the output
MatrixXd predict(MatrixXd X, MatrixXd w)
{ return X * w;}

string TestRegressionModel(int s, MatrixXd w, MatrixXd X_test, MatrixXd Y_test)
{

  int count = 0;
  string res;

  // making prediction
  MatrixXd pred = X_test * w;
  // cout << pred << endl;
  
  if (s==1 || s==2){

    MatrixXd diff = Y_test - pred;
    MatrixXd MSE = diff.transpose() * diff;

    res = "Loss " + to_string(MSE(0,0)/N_test);
  }
  else if (s==3){
    for (int i=0; i < N_test; i++)
    {
      if ((pred(i,0) < 0.5 && Y_test(i,0)==0) || (pred(i,0) >= 0.5 && Y_test(i,0) == 1))
      {
        count += 1;
      }
    }
    cout << "Count: " << count << endl;
    float accuracy = ((float) count/N_test * 100.0);
    res = to_string(accuracy) + " %";
  }

  return res;
}