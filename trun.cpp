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


#define SCALING_FACTOR 8192 // Precision of 13 bits
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;

// for medical dataset
int N = 1070;
int N_test = 268;
int d = 5;
int B = 2;
int NUM_EPOCHS = 10;

// function that takes float to unit64

MatrixXi64 floattouint64(MatrixXd A){

  // // same scaling method for both positive and negative floating points
  // A = A * SCALING_FACTOR;
  // MatrixXi64 res = A.cast<uint64_t>();

  // different scaling method for positive and negative floating points
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
        res(i,j) = (uint64_t) (pow(2,64) - a);
        //cout<< res(i,j) << " is negative"<<endl;
      }
    }
  } 

  return res;
}
// function that takes unit64 to float
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
        res(i,j) = pow(2,64) - (pow(2,64) - a)/factor;
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
  { cout<< "Epoch Number: "<< e+1;
    //cout<<" Progress: ";
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++) // for each nuber of iteration in an epoch
    { //cout<<"=";
      MatrixXd YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
      //cout<< "yhat: "<< endl << YY << endl;
      MatrixXd D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
      //cout<< "diff: "<< endl << D << endl;
      // Loss Computation
      MatrixXd loss = D.transpose() * D;
      MatrixXd delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      //cout<< "grad: " << endl << delta << endl;
      w = w - (delta /128); // w -= a/B * delta
      //cout<<"weights: "<< endl << w <<endl;
      //cout<<w<<endl;
      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  
  return w;
}

// Non-PP Linear Regression for integer inputs
MatrixXi idealLinearRegression(MatrixXi X, MatrixXi Y, MatrixXi w)
{
  // w -= a/|B| X^T .(X.w - Y)
  int t = (N * NUM_EPOCHS)/B; // E = 1
  for(int i = 0; i < t; i ++)
  {
    MatrixXi YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
    MatrixXi D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
    MatrixXi delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
    w = w - (delta / 128); // w -= a/B * delta
  }
  return w;
}


MatrixXi64 truncatedLinearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w) // ideal functionality
{
  // int N = 6; //6
  // int d = 2; //5
  // int B = 1; //3
  // int NUM_EPOCHS = 5; 

  // w -= a/|B| X^T .(X.w - Y)
  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1;
    //cout<<" Progress: ";
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++)
    { 
      //cout<<"=";
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
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  
  return w;
}


MatrixXd predict(MatrixXd X, MatrixXd Y, MatrixXd w)
{ MatrixXd pred = X * w;
  MatrixXd diff = pred - Y;
  MatrixXd loss = diff.transpose() * diff;
  cout<< "Test Loss: "<< loss(0,0)/X.rows() << endl;
  return pred;
}

MatrixXd predict(MatrixXd X, MatrixXd w)
{ return X * w;}

int main(){
  
  //==========================================
  // LOADING DATA:
  //==========================================
  cout<<"Reading Data:"<<endl;
  vector<vector<double> > X_train_input;   
  vector<double> Y_train_input;            
  
  // training data
  read_insurance_data("datasets/medical/insurance_train.csv", X_train_input, Y_train_input);

  MatrixXd X_train(N, d); 
  MatrixXd Y_train(N, 1); 

  for (int i = 0; i < N; i++)
  {
    X_train.row(i) = VectorXd::Map(&X_train_input[i][0], d)/10.0;
    Y_train.row(i) = VectorXd::Map(&Y_train_input[i],1)/10000.0;
  }

  vector<vector<double> > X_test_input;    
  vector<double> Y_test_input;             

  read_insurance_data("datasets/medical/insurance_test.csv", X_test_input, Y_test_input); 

  MatrixXd X_test(N_test, d); 
  MatrixXd Y_test(N_test, 1); 

  for (int i = 0; i < N_test; i++)
  {
    X_test.row(i) = VectorXd::Map(&X_test_input[i][0], d)/10.0;
    Y_test.row(i) = VectorXd::Map(&Y_test_input[i],1)/10000.0;
  }

  MatrixXd w1 = MatrixXd::Random(d,1);
  
  // Sanity Check data
  MatrixXd X2(6,2);
  X2 << 4,1,2,8,1,0,3,2,1,4,6,7;
  MatrixXd Y2(6,1);
  Y2 << 2,-14,1,-1,-7,-8;
  MatrixXd w2(2,1);
  w2 << 1.65921924,1.62628418;
  //cout << "Here is the matrix X2:\n" << X2 <<endl;
  //cout << "Here is the matrix Y2:\n" << Y2 <<endl;
  //==========================================


  //==========================================
  // MODEL TRAINING:
  //==========================================

  cout << "=============================="<<endl;
  cout << "IDEAL LINEAR REGRESSION (SGD):"<<endl;
  cout << "=============================="<<endl<<endl;
  MatrixXd ideal_w = idealLinearRegression(X_train,Y_train,w1);
  cout << "Final weights (under Ideal Functionality) are:\n" << ideal_w <<endl;

  //==========================================
  // MODEL PREDICTION:
  //==========================================

  // cout << endl << "==================================="<<endl;
  // cout << "PREDICTION (using trained weights):"<<endl;
  // cout << "==================================="<<endl<<endl;
  // MatrixXd pred = predict(X_test, Y_test, ideal_w);
  // //cout << pred <<endl;

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
  MatrixXi64 w_ = floattouint64(w1);


  MatrixXi64 new_w_ = truncatedLinearRegression(X_,Y_,w_);
  MatrixXd new_w_f = uint64tofloat(new_w_); // descaling

  cout << "Final weights (under truncated Functionality) are:\n" << new_w_f <<endl;
}