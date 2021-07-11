#include "read_data.hpp"
#include "defines.hpp"
#include "utils.hpp"
#include "linear_regression.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
====================================
Parameters:
- n: number of samples
- d: number of features
- B: batch size
- E: number of epochs
- t: number of iterations = n.E/B
====================================
Dimensions:
- X: (n,d)
- Y: (n,1)
- U: (n,d)
- V: (d,t)
- VV: (B,t)
- Z: (B,t)
- ZZ: (d,t)
====================================
*/


// ====================================
// Global Declarations: 
// ==================================== 

// for MNIST
//int N = 1000; // 10000
//int N_test = 100; // 1000
//int d = 784; //784
//int B = 128; //128
//int NUM_EPOCHS = 5; 

// for sanity check data
int N = 6; //6
int d = 2; //5
int B = 3; //3
int NUM_EPOCHS = 5;

// for medical dataset
//int N = 1070;
//int N_test = 268;
//int d = 5;
//int B = 128;
//int NUM_EPOCHS = 5;
// ====================================

int main(){

  //==========================================
  // Loadin and Pre processing data:
  //==========================================
/*
  cout<<"Reading Data:"<<endl;
  // loading mnist dataset: training and testing portion separately
  vector<vector<double> > X_train;   // dim: 60000 x 784, 60000 training samples with 784 features
  vector<double> Y_train;            // dim: 60000 x 1  , the true label of each training sample
  
  //read_insurance_data("datasets/medical/insurance_train.csv", X_train, Y_train); // for medical dataset
  read_data("datasets/mnist/mnist_train.csv", X_train, Y_train);             // for MNIST dataset
  

  MatrixXd X1(N, d); // 60000, 784
  MatrixXd Y1(N, 1); // 60000, 1

  for (int i = 0; i < N; i++)
  {
    X1.row(i) = VectorXd::Map(&X_train[i][0], d)/256.0;
    Y1.row(i) = VectorXd::Map(&Y_train[i],1)/10.0;
  }

  // cout << "Working: " << endl;

  vector<vector<double> > X_test;    // dim: 10000 x 784, 10000 testing samples with 784 features
  vector<double> Y_test;             // dim: 10000 x 1  , the true label of each testing sample


  //read_insurance_data("datasets/medical/insurance_test.csv", X_test, Y_test); // for medical dataset
  read_data("datasets/mnist/mnist_test.csv", X_test, Y_test);                  // for MNIST dataset

  MatrixXd X1_test(N_test, d); // 1000, 784
  MatrixXd Y1_test(N_test, 1); // 1000, 1

  for (int i = 0; i < N_test; i++)
  {
    X1_test.row(i) = VectorXd::Map(&X_test[i][0], d)/256.0;
    Y1_test.row(i) = VectorXd::Map(&Y_test[i],1)/10.0;
  }

  MatrixXd w1 = MatrixXd::Random(d,1);
*/ 
  //cout << "Here is Matrix w1:\n" << w1 <<endl;
  //==========================================

  //==========================================
  // Sanity Check data:
  //==========================================
  MatrixXd X2(6,2);
  X2 << 4,1,-2.4,8,1,0.11,3,2.3,1,4,6.6,7.32;
  MatrixXd Y2(6,1);
  Y2 << 2,-14,1,-1,-7,-8;
  MatrixXd w2(2,1);
  w2 << 1.65921924,1.62628418;
  //==========================================


  //==========================================
  // MODEL TRAINING:
  //==========================================

  //cout<<"Initial weights: "<< endl << w2 <<endl<<endl;

  cout << "=============================="<<endl;
  cout << "FLOATING LINEAR REGRESSION (SGD):"<<endl;
  cout << "=============================="<<endl<<endl;

  MatrixXd ideal_w = idealLinearRegression(X2,Y2,w2);
  //cout << "Final weights (under Ideal Functionality) are:\n" << ideal_w << endl << endl;
  MatrixXd new_w = linearRegression(X2,Y2,w2);
  //cout << "Final weights (under Privacy Preserving) are:\n" << new_w << endl << endl;

  cout << "=============================="<<endl;
  cout << "UINT-64 LINEAR REGRESSION (SGD):"<<endl;
  cout << "=============================="<<endl<<endl;

  MatrixXi64 X_ = floattouint64(X2); // double to uint_64
  MatrixXi64 Y_ = floattouint64(Y2); // double to uint_64
  MatrixXi64 w_ = floattouint64(w2); // double to uint_64

  MatrixXi64 new_w_ = idealLinearRegression(X_,Y_,w_);
  MatrixXd new_w_f = uint64tofloat(new_w_); // descaling

  //cout<<"Final weights (under Ideal Functionality)  are:\n "<< new_w_f << endl << endl;

  new_w_ = linearRegression(X_,Y_,w_);
  new_w_f = uint64tofloat(new_w_); // descaling

  //cout<<"Final weights (under Privacy Preserving) are:\n "<< new_w_f << endl;

}

