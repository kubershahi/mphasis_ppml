#include "read_data.hpp"                  // functions to read data
#include "defines.hpp"                    // global definitions
#include "utils.hpp"                      // utility functions
#include "linear_regression.hpp"          // linear regression for pp and non-pp settings

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
- VV (V'): (B,t)
- Z: (B,t)
- ZZ(Z'): (d,t)
====================================
*/

// ====================================
// Global Declarations: 
// ==================================== 

// for sanity check data
int N = 6; //6
int N_test = 6;
int d = 2; //5
int B = 3; //3
int NUM_EPOCHS = 3;

// ====================================

int main(){

  //==========================================
  // Loading and Pre-Processing data:
  //==========================================

  cout<<"Select Dataset (enter corresponding digit):"<<endl;
  cout<<"\t [1] MNIST"<<endl;
  cout<<"\t [2] Medical Insurance"<<endl;
  cout<<"\t [3] Binary MNIST"<<endl;
  cout<<"\t [4] Sanity Check"<<endl;
  int selection = 0;
  cout<<"Enter selection: ";
  cin>>selection;

  MatrixXd X,Y,w,X_testdata,Y_testdata;

  if (selection == 1){ // loading data for MNIST dataset
  
  :: N = 60000; // 10000
  :: N_test = 10000; // 1000
  :: d = 784; //784
  :: B = 128; //128
  :: NUM_EPOCHS = 4;

  cout<<"Reading Data:"<<endl;
  
  vector<vector<double> > X_train;   // dim: 60000 x 784, 60000 training samples with 784 features
  vector<double> Y_train;            // dim: 60000 x 1  , the true label of each training sample
  
  read_data("datasets/mnist/mnist_train.csv", X_train, Y_train);

  MatrixXd X1(N, d); // 60000, 784
  MatrixXd Y1(N, 1); // 60000, 1

  for (int i = 0; i < N; i++)
  {
    X1.row(i) = VectorXd::Map(&X_train[i][0], d)/256.0;
    Y1.row(i) = VectorXd::Map(&Y_train[i],1)/10.0;
  }

  vector<vector<double> > X_test;    // dim: 10000 x 784, 10000 testing samples with 784 features
  vector<double> Y_test;             // dim: 10000 x 1  , the true label of each testing sample

  read_data("datasets/mnist/mnist_test.csv", X_test, Y_test);                  // for MNIST dataset

  MatrixXd X1_test(N_test, d); // 1000, 784
  MatrixXd Y1_test(N_test, 1); // 1000, 1

  for (int i = 0; i < N_test; i++)
  {
    X1_test.row(i) = VectorXd::Map(&X_test[i][0], d)/256.0;
    Y1_test.row(i) = VectorXd::Map(&Y_test[i],1)/10.0;
  }

  MatrixXd w1 = MatrixXd::Random(d,1);

  X = X1;
  Y = Y1;
  w = w1;

  X_testdata = X1_test;
  Y_testdata = Y1_test;

  }
  else if (selection == 2){ // loading data for Medical dataset

  :: N = 1070; // 1070
  :: N_test = 268; // 268
  :: d = 5; // 5
  :: B = 1; //2
  :: NUM_EPOCHS = 90; //100

  cout<<"Reading Data:"<<endl;
  
  vector<vector<double> > X_train;   // dim: 1070 x 5, 1070 training samples with 5 features
  vector<double> Y_train;            // dim: 1070 x 1, the true label of each training sample
  
  read_insurance_data("datasets/medical/insurance_train.csv", X_train, Y_train); 

  MatrixXd X3(N, d); // 1070, 5
  MatrixXd Y3(N, 1); // 268, 1

  for (int i = 0; i < N; i++)
  {
    X3.row(i) = VectorXd::Map(&X_train[i][0], d)/100.0;
    Y3.row(i) = VectorXd::Map(&Y_train[i],1)/10000.0;
  }

  vector<vector<double> > X_test;    // dim: 268 x 5, 268 testing samples with 5 features
  vector<double> Y_test;             // dim: 368 x 1, the true label of each testing sample

  read_insurance_data("datasets/medical/insurance_test.csv", X_test, Y_test);

  MatrixXd X3_test(N_test, d); // 1000, 784
  MatrixXd Y3_test(N_test, 1); // 1000, 1

  for (int i = 0; i < N_test; i++)
  {
    X3_test.row(i) = VectorXd::Map(&X_test[i][0], d)/100.0;
    Y3_test.row(i) = VectorXd::Map(&Y_test[i],1)/10000.0;
  }

  MatrixXd w3 = MatrixXd::Random(d,1);

  X = X3;
  Y = Y3;
  w = w3;
  X_testdata = X3_test;
  Y_testdata = Y3_test;

  }
  else if (selection == 3){ // loading Binary MNIST dataset

  :: N = 60000; // 10000
  :: N_test = 10000; // 1000
  :: d = 784; //784
  :: B = 128; //128
  :: NUM_EPOCHS = 4;

  cout<<"Reading Data:"<<endl;
  
  vector<vector<double> > X_train;   // dim: 60000 x 784, 60000 training samples with 784 features
  vector<double> Y_train;            // dim: 60000 x 1  , the true label of each training sample
  
  read_data("datasets/binary_mnist/mnist_train.csv", X_train, Y_train);

  MatrixXd X1(N, d); // 60000, 784
  MatrixXd Y1(N, 1); // 60000, 1

  for (int i = 0; i < N; i++)
  {
    X1.row(i) = VectorXd::Map(&X_train[i][0], d)/256.0;
    Y1.row(i) = VectorXd::Map(&Y_train[i],1);
  }

  vector<vector<double> > X_test;    // dim: 10000 x 784, 10000 testing samples with 784 features
  vector<double> Y_test;             // dim: 10000 x 1  , the true label of each testing sample

  read_data("datasets/binary_mnist/mnist_test.csv", X_test, Y_test);                  // for MNIST dataset

  MatrixXd X1_test(N_test, d); // 1000, 784
  MatrixXd Y1_test(N_test, 1); // 1000, 1

  for (int i = 0; i < N_test; i++)
  {
    X1_test.row(i) = VectorXd::Map(&X_test[i][0], d)/256.0;
    Y1_test.row(i) = VectorXd::Map(&Y_test[i],1);
  }

  MatrixXd w1 = MatrixXd::Random(d,1);

  X = X1;
  Y = Y1;
  w = w1;
  X_testdata = X1_test;
  Y_testdata = Y1_test;

  }
  else{ // Sanity Check Data:

  MatrixXd X2(6,2);
  X2 << 4,1,-2.4,8,1,0.11,3,2.3,1,4,6.6,7.32;
  MatrixXd Y2(6,1);
  Y2 << 2,-14,1,-1,-7,-8;
  MatrixXd w2(2,1);
  w2 << 1.65921924,1.62628418;

  X = X2;
  Y = Y2;
  w = w2;

  }


  //==========================================
  // MODEL TRAINING:
  //==========================================

  //cout<<"Initial weights: "<< endl << w <<endl<<endl;

  cout << endl;
  cout << "================================================"<<endl;
  cout << "NON-PP LINEAR REGRESSION (Floating Point Inputs):"<<endl;
  cout << "================================================"<<endl<<endl;

  MatrixXd ideal_w = idealLinearRegression(X,Y,w);
  // cout << "Final weights (under NON-PP Functionality) are:\n" << ideal_w << endl;
  
  cout << endl;
  cout << "Testing Regression Model: " << TestRegressionModel(selection, ideal_w, X_testdata, Y_testdata) << endl;
  
  cout << endl;
  cout << "====================================="<<endl;
  cout << "PP LINEAR REGRESSION (UINT-64 Inputs):"<<endl;
  cout << "====================================="<<endl<<endl;

  // conerting doubles to 64-bit integers
  MatrixXi64 X_ = floattouint64(X); // double to uint_64
  MatrixXi64 Y_ = floattouint64(Y); // double to uint_64
  MatrixXi64 w_ = floattouint64(w); // double to uint_64

  MatrixXi64 pp_w_i = linearRegression(X_,Y_,w_);
  MatrixXd pp_w = uint64tofloat(pp_w_i); // descaling

  // cout<<"Final weights (under Privacy Preserving functionality) are:\n "<< pp_w << endl;

  cout << endl;
  cout << "Testing Regression Model: " << TestRegressionModel(selection, pp_w, X_testdata, Y_testdata) << endl;

}