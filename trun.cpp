/*
Testing Truncation code 

idealLinearRegression: simple linear regression

truncatedLinearRegression: simple linear regression with truncation. Change floating points 
to integer and train on the integers
*/
#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include "read_data.hpp"

using namespace std;
using namespace Eigen;


#define SCALING_FACTOR 262144   // Precision of 16 bits
/*

2^16 = 65536
2^18 = 262144
2^20 = 1048576 
2^24 = 16777216
2^28 = 268435456

*/
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;

// for medical dataset
int N = 1070;
int N_test = 268;
int d = 5;
int B = 2;
int NUM_EPOCHS = 10;


// function that takes float to unit64
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


// function that takes uint64 to float
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


// function that truncates by a specified factor
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
        res(i,j) = (uint64_t) pow(2,64) - ((uint64_t) pow(2,64) - a)/factor;
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


// Non-PP Linear Regression for floating point inputs
MatrixXd idealLinearRegression(MatrixXd X, MatrixXd Y, MatrixXd w) // ideal functionality
{ 

  for(int e = 0; e < NUM_EPOCHS; e ++) // for each number of epochs
  { cout<< "Epoch Number: "<< e+1<< endl;
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++) // for each nuber of iteration in an epoch
    { 
      MatrixXd YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
      // cout<< "y_hat: "<< YY << endl;

      MatrixXd D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
      // cout<< "diff: "<< D << endl;

      MatrixXd loss = D.transpose() * D;

      MatrixXd delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      // cout<< "grad: " << delta << endl;

      w = w - (delta /128); // w -= a/B * delta
      // cout<<"weights: "<< endl << w <<endl;
      //cout<<w<<endl;
      epoch_loss += loss(0,0);
    }
    cout<<  "Loss: "<< epoch_loss/N << endl;
  }
  
  return w;
}


// function for truncated linear regression
MatrixXi64 truncatedLinearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w) // ideal functionality
{
  // w -= a/|B| X^T .(X.w - Y)
  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1<<endl;
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++)
    { 
      MatrixXi64 YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
      
      //truncation:
      //YY /= SCALING_FACTOR;
      YY = truncate(YY, SCALING_FACTOR);
      //test
      //MatrixXd YYtest = uint64tofloat(YY); // descaling
      //cout<< "yhat: "<< endl << YYtest << endl;
      MatrixXi64 D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
      //test
      //MatrixXd Dtest = uint64tofloat(D);// descaling
      //cout<< "diff: "<< endl << Dtest << endl;
      
      // Loss Computation
      MatrixXd loss = uint64tofloat(D).transpose() * uint64tofloat(D);

      MatrixXi64 delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      //cout<< "grad_raw: " << endl << delta << endl;

      //truncation:
      //delta /= SCALING_FACTOR;
      delta = truncate(delta, SCALING_FACTOR);
      //test
      //MatrixXd gradtest = uint64tofloat(delta);// descaling
      //cout<< "grad: " << endl << gradtest << endl;
      delta = truncate(delta, 128); // eta/B * delta, eta/B = 128 = 2^7 
      w = w - delta; // w -= a/B * delta
      //cout<<"weights: "<< endl << uint64tofloat(w) <<endl;
      epoch_loss += loss(0,0);
    }
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  return w;
}

int main(){


  // loading data

  cout<<"Reading Data:"<<endl;
  // loading training data
  vector<vector<double> > X_train_input;   // dim: 60000 x 784
  vector<double> Y_train_input;            // dim: 60000 x 1  
  
  read_insurance_data("datasets/medical/insurance_train.csv", X_train_input, Y_train_input); // for medical dataset
  // read_data("datasets/mnist/mnist_train.csv", X_train_input, Y_train_input);  

  MatrixXd X_train(N, d); // 60000, 784
  MatrixXd Y_train(N, 1); // 60000, 1  

  for (int i = 0; i < N; i++)
  {
    X_train.row(i) = VectorXd::Map(&X_train_input[i][0], d)/100.0;
    Y_train.row(i) = VectorXd::Map(&Y_train_input[i],1);
  }
  
  // // Validation single example
  // MatrixXd X_val(1,5);
  // X_val << 18.0, 1.0,	33.77, 1.0, 0.0;
  // MatrixXd Y_val(1,1);
  // Y_val << 1725.5523;
  // //cout << "Here is the matrix X_val:\n" << X_val <<endl;
  // //cout << "Here is the matrix Y_val:\n" << Y_val <<endl;

  MatrixXd w = MatrixXd::Random(d,1);
  cout <<endl << "Here is the weight matrix (raw):\n" << w <<endl;

  MatrixXi64 w_i = floattouint64(w);
  MatrixXd w_f = uint64tofloat(w_i);

  cout <<endl << "Here is the matrix matrix (mapped):\n" << w_f <<endl;
  //==========================================

  //==========================================
  // MODEL TRAINING:
  //==========================================

  cout << "=============================="<<endl;
  cout << "IDEAL LINEAR REGRESSION (SGD):"<<endl;
  cout << "=============================="<<endl<<endl;
  MatrixXd ideal_w = idealLinearRegression(X_train,Y_train,w);
  cout << endl << "Final weights (under Ideal Functionality) are:\n" << ideal_w <<endl<<endl;

  cout << "=============================="<<endl;
  cout << "Truncated LINEAR REGRESSION (SGD):"<<endl;
  cout << "=============================="<<endl<<endl;

  // X2 = X2 * SCALING_FACTOR; // double to uint_64
  // Y2 = Y2 * SCALING_FACTOR; // double to uint_64
  // //w1 = MatrixXd::Random(d,1);
  // w2 = w2 * SCALING_FACTOR; // double to uint_64

  // MatrixXi64 X_ = X2.cast<uint64_t>();
  // MatrixXi64 Y_ = Y2.cast<uint64_t>();
  // MatrixXi64 w_ = w2.cast<uint64_t>();

  MatrixXi64 X_ = floattouint64(X_train);
  MatrixXi64 Y_ = floattouint64(Y_train);
  MatrixXi64 w_ = floattouint64(w);

  // cout << "Here is the matrix w_:\n" << w_ <<endl;

  MatrixXi64 trun_w = truncatedLinearRegression(X_,Y_,w_);
  // cout << "Final weights before descaling are:\n" << new_w_ <<endl<<endl;
  
  MatrixXd trun_w_f = uint64tofloat(trun_w); // descaling
  cout << endl << "Final weights (under truncated Functionality) are:\n" << trun_w_f <<endl<<endl;
}