#include <iostream>
#include <Eigen/Dense>
#include "read_data.hpp"

using namespace std;
using namespace Eigen;


#define SCALING_FACTOR 8192 // Precision of 13 bits
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;

/*
// ==================================== 
Goals:
// ==================================== 
- Additive Secret Sharing [done]
- Adding shares [done]
- Multiplying shares using triplets [done]
- Online Phase of Linear Regression [done]
- Ideal Functionality (Linear Regression) [done]
- Batchwise SGD [done]
- Use NetIO for socket communication []
- Truncation (later)
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
int N = 10000; //6
int N_test = 1000;
int d = 784; //5
int B = 128; //3
int NUM_EPOCHS = 5; // change; shuffle order

// for sanity check data
//int N = 6; //6
//int N_test = 6;
//int d = 2; //5
//int B = 3; //3
//int NUM_EPOCHS = 5; // change; shuffle order

// for medical dataset
// int N = 1078;
// int N_test = 268;
// int d = 5;
// int B = 2;
// int NUM_EPOCHS = 100;
// ====================================


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

// For floating point inputs
MatrixXd rec(MatrixXd A, MatrixXd B)
{
  return A + B;
}

// =====================================================
// Other Helper Functions for Floating Point Arithmetic: 
// ===================================================== 

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

// ====================================
// Secure Multiplication using Beavers' Triplets: 
// ==================================== 
// For integer inputs
MatrixXi mult(int i, MatrixXi A, MatrixXi B, MatrixXi E, MatrixXi F, MatrixXi Z)
{ 
  //MatrixXi pp = E*F + U*F + E*V + Z;
  //MatrixXi pp = -E*F + X*F + E*w + Z;
  //MatrixXi pp = E*F + X*F + E*w + Z; //--> doesn't work (SecureML)
	MatrixXi product = MatrixXi::Random(Z.rows(),Z.cols());
	if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
	else if (i == 0) product = (A * F) + (E * B) + Z;
	
	return product;
}

// For floating point inputs
MatrixXd mult(int i, MatrixXd A, MatrixXd B, MatrixXd E, MatrixXd F, MatrixXd Z)
{ 
  //MatrixXf pp = E*F + U*F + E*V + Z;
  //MatrixXf pp = -E*F + X*F + E*w + Z;
  //MatrixXf pp = E*F + X*F + E*w + Z; //--> doesn't work (SecureML)
  MatrixXd product = MatrixXd::Random(Z.rows(),Z.cols());
  if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
  else if (i == 0) product = (A * F) + (E * B) + Z;
  
  return product;
}

// ==================================== 
// ==================================== 

// Linear Regression for Integer Input only
MatrixXi linearRegression(MatrixXi X, MatrixXi Y, MatrixXi w)
{ 
  // ===========================
  // Additive Secret Sharing
  // ===========================
  MatrixXi shares[2];

  share(X, shares); // training data shares
  MatrixXi X_0 = shares[0];
  MatrixXi X_1 = shares[1];
  share(Y, shares); // label shares
  MatrixXi Y_0 = shares[0];
  MatrixXi Y_1 = shares[1];
  share(w, shares); // weight shares
  MatrixXi w_0 = shares[0];
  MatrixXi w_1 = shares[1];

  // ===========================
  // Triplet Generation (Offline Phase)
  // ===========================
  MatrixXi U = MatrixXi::Random(X.rows(),X.cols()) / 10000000; // masks X_i
  share(U, shares);
  MatrixXi U_0 = shares[0]; 
  MatrixXi U_1 = shares[1];

  MatrixXi E_0 = X_0 - U_0;
  MatrixXi E_1 = X_1 - U_1;
  MatrixXi E = rec(E_0, E_1); // masked X_i

  int t = (N * NUM_EPOCHS)/B; // E = 1

  MatrixXi V = MatrixXi::Random(d, t) / 10000000; // masks w_i
  share(V, shares);
  MatrixXi V_0 = shares[0]; 
  MatrixXi V_1 = shares[1];

  //MatrixXi Z = U * V; // third triplet for multiplication -> change for batchwise SGD
  MatrixXi Z = MatrixXi::Zero(B,t); 
  for (int z = 0; z < Z.cols(); z++)
  {
    Z.col(z) = U.block(z * B,0,B,U.cols()) * V.col(z);
  }
  share(Z, shares);
  MatrixXi Z_0 = shares[0];
  MatrixXi Z_1 = shares[1];

  MatrixXi VV = MatrixXi::Random(B,t) / 10000000; // masks D_i
  share(VV, shares);
  MatrixXi VV_0 = shares[0]; 
  MatrixXi VV_1 = shares[1];

  //MatrixXi ZZ = U.transpose() * VV; // third triplet for multiplication -> change for batchwise SGD
  MatrixXi ZZ = MatrixXi::Zero(d,t);
  for (int z = 0; z < ZZ.cols(); z++)
  {
    ZZ.col(z) = U.transpose().block(0,z * B,U.cols(),B) * VV.col(z);
  }
  share(ZZ, shares);
  MatrixXi ZZ_0 = shares[0];
  MatrixXi ZZ_1 = shares[1];

  // ===========================
  // Online Phase
  // ===========================
  for(int j = 0; j < t; j ++)
  {

    MatrixXi F_0 = w_0 - V_0.col(j);
    MatrixXi F_1 = w_1 - V_1.col(j);
    MatrixXi F = rec(F_0, F_1);

    // YY = X_B_j.w
    MatrixXi YY_0 = mult(0, X_0.block(B * j,0,B,X.cols()), w_0, E.block(B * j,0,B,X.cols()), F, Z_0.col(j));
    MatrixXi YY_1 = mult(1, X_1.block(B * j,0,B,X.cols()), w_1, E.block(B * j,0,B,X.cols()), F, Z_1.col(j));

    // D = X.w - Y
    MatrixXi D_0 = YY_0 - Y_0.block(B * j,0,B,Y.cols());
    MatrixXi D_1 = YY_1 - Y_1.block(B * j,0,B,Y.cols());

    MatrixXi FF_0 = D_0 - VV_0.col(j);
    MatrixXi FF_1 = D_1 - VV_1.col(j);
    MatrixXi FF = rec(FF_0, FF_1);

    // delta = X^T(X.w - Y)
    MatrixXi delta_0 = mult(0, X_0.transpose().block(0,B * j,X.cols(),B), D_0, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_0.col(j));
    MatrixXi delta_1 = mult(1, X_1.transpose().block(0,B * j,X.cols(),B), D_1, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_1.col(j));

    // Truncation
    // =========================== 
    // To be added
    // =========================== 

    // delta = X^T(X.w - Y)
    w_0 = w_0 - (delta_0 / (B * 100)); // alpha/b = 2^-7 = 128; used in paper
    w_1 = w_1 - (delta_1 / (B * 100)); // alpha/b = 2^-7 = 128; used in paper
  }

  return rec(w_0, w_1);
}

// Linear Regression for floating point input
MatrixXd linearRegression(MatrixXd X, MatrixXd Y, MatrixXd w)
{ 
  // ===========================
  // Additive Secret Sharing
  // ===========================
  MatrixXd shares[2];

  share(X, shares); // training data shares
  MatrixXd X_0 = shares[0];
  MatrixXd X_1 = shares[1];
  share(Y, shares); // label shares
  MatrixXd Y_0 = shares[0];
  MatrixXd Y_1 = shares[1];
  share(w, shares); // weight shares
  MatrixXd w_0 = shares[0];
  MatrixXd w_1 = shares[1];
  // ===========================
  // Triplet Generation (Offline Phase)
  // ===========================
  MatrixXd U = MatrixXd::Random(X.rows(),X.cols()); // masks X_i
  share(U, shares);
  MatrixXd U_0 = shares[0]; 
  MatrixXd U_1 = shares[1];

  MatrixXd E_0 = X_0 - U_0;
  MatrixXd E_1 = X_1 - U_1;
  MatrixXd E = rec(E_0, E_1); // masked X_i

  int t = (N)/B; // E = 1

  MatrixXd V = MatrixXd::Random(d, t); // masks w_i
  share(V, shares);
  MatrixXd V_0 = shares[0]; 
  MatrixXd V_1 = shares[1];

  //MatrixXd Z = U * V; // third triplet for multiplication -> change for batchwise SGD
  MatrixXd Z = MatrixXd::Zero(B,t); 
  for (int z = 0; z < Z.cols(); z++)
  {
    Z.col(z) = U.block(z * B,0,B,U.cols()) * V.col(z);
  }
  share(Z, shares);
  MatrixXd Z_0 = shares[0];
  MatrixXd Z_1 = shares[1];

  MatrixXd VV = MatrixXd::Random(B,t); // masks D_i
  share(VV, shares);
  MatrixXd VV_0 = shares[0]; 
  MatrixXd VV_1 = shares[1];

  //MatrixXd ZZ = U.transpose() * VV; // third triplet for multiplication -> change for batchwise SGD
  MatrixXd ZZ = MatrixXd::Zero(d,t);
  for (int z = 0; z < ZZ.cols(); z++)
  {
    ZZ.col(z) = U.transpose().block(0,z * B,U.cols(),B) * VV.col(z);
  }
  share(ZZ, shares);
  MatrixXd ZZ_0 = shares[0];
  MatrixXd ZZ_1 = shares[1];

  // ===========================
  // Online Phase
  // ===========================
  for(int e = 0; e < NUM_EPOCHS; e++)
  { cout<< "Epoch Number: "<< e+1;
    cout<<" Progress: ";
    float epoch_loss = 0.0;

    for(int j = 0; j < int(N/B); j++)
    {
      cout<<"=";
      MatrixXd F_0 = w_0 - V_0.col(j);
      MatrixXd F_1 = w_1 - V_1.col(j);
      MatrixXd F = rec(F_0, F_1);

      // YY = X_B_j.w
      MatrixXd YY_0 = mult(0, X_0.block(B * j,0,B,X.cols()), w_0, E.block(B * j,0,B,X.cols()), F, Z_0.col(j));
      MatrixXd YY_1 = mult(1, X_1.block(B * j,0,B,X.cols()), w_1, E.block(B * j,0,B,X.cols()), F, Z_1.col(j));

      // D = X.w - Y
      MatrixXd D_0 = YY_0 - Y_0.block(B * j,0,B,Y.cols());
      MatrixXd D_1 = YY_1 - Y_1.block(B * j,0,B,Y.cols());

      // Loss Computation (for testing only)
      MatrixXd D = rec(D_0, D_1);
      MatrixXd loss = D.transpose() * D;

      MatrixXd FF_0 = D_0 - VV_0.col(j);
      MatrixXd FF_1 = D_1 - VV_1.col(j);
      MatrixXd FF = rec(FF_0, FF_1);

      // delta = X^T(X.w - Y)
      MatrixXd delta_0 = mult(0, X_0.transpose().block(0,B * j,X.cols(),B), D_0, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_0.col(j));
      MatrixXd delta_1 = mult(1, X_1.transpose().block(0,B * j,X.cols(),B), D_1, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_1.col(j));

      // Truncation
      // =========================== 
      // To be added
      // =========================== 

      // delta = X^T(X.w - Y)
      w_0 = w_0 - (delta_0 / (B * 100)); // alpha/b = 2^-7 = 128; used in paper
      w_1 = w_1 - (delta_1 / (B * 100)); // alpha/b = 2^-7 = 128; used in paper

      //cout<< rec(w_0, w_1) <<endl;
      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  

  return rec(w_0, w_1);
}

// Non-PP Linear Regression for floating point inputs
MatrixXd idealLinearRegression(MatrixXd X, MatrixXd Y, MatrixXd w) // ideal functionality
{
  // w -= a/|B| X^T .(X.w - Y)
  //int t = (N * NUM_EPOCHS)/B; // E = 1
  //float eta = 0.01;
  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1;
    cout<<" Progress: ";
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++)
    { cout<<"=";
      MatrixXd YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
      MatrixXd D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
      // Loss Computation
      MatrixXd loss = D.transpose() * D;
      MatrixXd delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      w = w - (delta / (B * 100)); // w -= a/B * delta
      //cout<<w<<endl;
      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  
  return w;
}

MatrixXi idealLinearRegression(MatrixXi X, MatrixXi Y, MatrixXi w) // ideal functionality
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


MatrixXi64 idealFloatingLinearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w) // ideal functionality
{
  // w -= a/|B| X^T .(X.w - Y)
  //int t = (N * NUM_EPOCHS)/B; // E = 1
  //float eta = 0.01;

  int N = 6; //6
  int d = 2; //5
  int B = 3; //3
  int NUM_EPOCHS = 5; 

  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1;
    cout<<" Progress: ";
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++)
    { 
      cout<<"=";
      MatrixXi64 YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w

      //truncation:
      //YY /= SCALING_FACTOR;
      YY = truncate(YY, SCALING_FACTOR);

      //test
      MatrixXd YYtest = uint64tofloat(YY); // descaling
      //cout<< "yhat: "<< endl << YYtest << endl;

      MatrixXi64 D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i

      //test
      MatrixXd Dtest = uint64tofloat(D);// descaling
      //cout<< "diff: "<< endl << Dtest << endl;
      
      // Loss Computation
      MatrixXd loss = uint64tofloat(D).transpose() * uint64tofloat(D);

      MatrixXi64 delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      //cout<< "grad_raw: " << endl << delta << endl;

      //truncation:
      //delta /= SCALING_FACTOR;
      delta = truncate(delta, SCALING_FACTOR);
      //test
      MatrixXd gradtest = uint64tofloat(delta);// descaling
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
  // Loadin and Pre processing data:
  //==========================================
  cout<<"Reading Data:"<<endl;
  // loading mnist dataset: training and testing portion separately
  vector<vector<double> > X_train;   // dim: 60000 x 784, 60000 training samples with 784 features
  vector<double> Y_train;            // dim: 60000 x 1  , the true label of each training sample
  
  // read_insurance_data("datasets/medical/insurance_train.csv", X_train, Y_train); // for medical dataset
  read_data("datasets/mnist/mnist_train.csv", X_train, Y_train);             // for MNIST dataset
  
  // for (int i =0; i <10; i++){
  //   cout << X_train[i][0] << " "  << X_train[i][1] << " " << X_train[i][2] << " " << X_train[i][3] << " " << X_train[i][4] << " " << Y_train[i] << endl;
  // }

  MatrixXd X1(N, d); // 60000, 784
  MatrixXd Y1(N, 1); // 60000, 1

  for (int i = 0; i < N; i++)
  {
    X1.row(i) = VectorXd::Map(&X_train[i][0], d)/255.0;
    Y1.row(i) = VectorXd::Map(&Y_train[i],1)/10.0;
  }

  cout << "Working: " << endl;

  vector<vector<double> > X_test;    // dim: 10000 x 784, 10000 testing samples with 784 features
  vector<double> Y_test;             // dim: 10000 x 1  , the true label of each testing sample


  // read_insurance_data("datasets/medical/insurance_test.csv", X_test, Y_test); // for medical dataset
  read_data("datasets/mnist/mnist_test.csv", X_test, Y_test);                  // for MNIST dataset

  MatrixXd X1_test(N_test, d); // 1000, 784
  MatrixXd Y1_test(N_test, 1); // 1000, 1

  for (int i = 0; i < N_test; i++)
  {
    X1_test.row(i) = VectorXd::Map(&X_test[i][0], d)/255.0;
    Y1_test.row(i) = VectorXd::Map(&Y_test[i],1)/10.0;
  }

  MatrixXd w1 = MatrixXd::Random(d,1);
  //cout << "Here is Matrix w1:\n" << w1 <<endl;
  //==========================================

  //==========================================
  // Random data:
  //==========================================

  //MatrixXi X = MatrixXi::Random(N,d) / 10000000; // n = 6, d = 5, training samples
  //cout << "Here is the matrix X:\n" << X <<endl;
  //MatrixXi Y = MatrixXi::Random(N,1) / 10000000; // labels
  //cout << "Here is the matrix Y:\n" << Y <<endl;
  //MatrixXi w = MatrixXi::Random(d,1) / 100000000;

  //MatrixXi new_w = linearRegression(X,Y,w);
  //cout << "Final weights (under Privacy Preserving) are:\n" << new_w <<endl;
  //==========================================


  //==========================================
  // Sanity Check data:
  //==========================================
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
  MatrixXd ideal_w = idealLinearRegression(X1,Y1,w1);
  //cout << "Final weights (under Ideal Functionality) are:\n" << ideal_w <<endl;
  cout << endl << "==========================================="<<endl;
  cout << "PRIVACY-PRESERVING LINEAR REGRESSION (SGD):"<<endl;
  cout << "==========================================="<<endl<<endl;
  MatrixXd new_w = linearRegression(X1,Y1,w1);
  //cout << "Final weights (under Privacy Preserving) are:\n" << new_w <<endl;



  //==========================================
  // MODEL PREDICTION:
  //==========================================
  /*
  cout << endl << "==================================="<<endl;
  cout << "PREDICTION (using trained weights):"<<endl;
  cout << "==================================="<<endl<<endl;
  MatrixXd pred = predict(X1_test, Y1_test, new_w);
  //cout << pred <<endl;
*/
  /*
  cout << endl << "Single example predictions: " << endl;
<<<<<<< HEAD
  for (int k = 500; k < 701; k += 100){
=======

  for (int k = 100; k < 200; k += 15){
>>>>>>> ad67b80fbdcd230b649535f71b8da79e42e8a2d7
    cout << "True Label: " << Y1_test.row(k) << endl;
    MatrixXd pred_i = predict(X1_test.row(k), ideal_w);
    cout << "Ideal Prediction: " << pred_i <<endl;
    MatrixXd pred_p = predict(X1_test.row(k), new_w);
    cout << "PP Prediction: " << pred_p <<endl;
  }
  */

  cout<<"============================"<<endl<<"============================"<<endl;

  X2 = X2 * SCALING_FACTOR; // double to uint_64
  Y2 = Y2 * SCALING_FACTOR; // double to uint_64
  //w1 = MatrixXd::Random(d,1);
  w2 = w2 * SCALING_FACTOR; // double to uint_64

  MatrixXi64 X_ = X2.cast<uint64_t>();
  MatrixXi64 Y_ = Y2.cast<uint64_t>();
  MatrixXi64 w_ = w2.cast<uint64_t>();

  MatrixXi64 new_w_ = idealFloatingLinearRegression(X_,Y_,w_);
  MatrixXd new_w_f = uint64tofloat(new_w_); // descaling

  //cout<<"Final weights are: "<< new_w_f <<endl;

}

void verify()
{
  /*

  MatrixXi shares[2];

  share(X, shares);
  MatrixXi X_0 = shares[0];
  MatrixXi X_1 = shares[1];
  share(Y, shares);
  MatrixXi Y_0 = shares[0];
  MatrixXi Y_1 = shares[1];


  //cout << "Here is the matrix X_0 + X_1:\n" << X_0 + X_1 <<endl;
  //cout << "Here is the matrix Y_0 + Y_1:\n" << Y_0 + Y_1 <<endl;

  // Multiplication
  MatrixXi w_0 = MatrixXi::Random(5,1) / 10000000;
  MatrixXi w_1 = MatrixXi::Random(5,1) / 10000000;
  MatrixXi w = rec(w_0, w_1);
  //cout << "Here is the matrix w_0 + w_1:\n" << w_0 + w_1 <<endl;
  
  //Masking

  MatrixXi U = MatrixXi::Random(X.rows(),X.cols()) / 10000;
  share(U, shares);
  MatrixXi U_0 = shares[0]; 
  MatrixXi U_1 = shares[1];

  MatrixXi E_0 = X_0 - U_0;
  MatrixXi E_1 = X_1 - U_1;
  MatrixXi E = rec(E_0, E_1);

  cout << "Here is the matrix U + E:\n" << U + E <<endl;
  //cout<< "Dimensions of X is: "<<X.rows()<<" "<<X.cols()<<endl;

  MatrixXi V = MatrixXi::Random(w.rows(),w.cols()) / 10000;
  share(V, shares);
  MatrixXi V_0 = shares[0]; 
  MatrixXi V_1 = shares[1];

  MatrixXi F_0 = w_0 - V_0;
  MatrixXi F_1 = w_1 - V_1;
  MatrixXi F = rec(F_0, F_1);

  MatrixXi Z = U * V;
  share(Z, shares);
  MatrixXi Z_0 = shares[0];
  MatrixXi Z_1 = shares[1]; 

  MatrixXi prod_0 = mult(0, X_0, w_0, E, F, Z_0);
  MatrixXi prod_1 = mult(1, X_1, w_1, E, F, Z_1);

  MatrixXi prod_test = X * w;
  cout << "Here is the required product:\n" << prod_test <<endl;
  MatrixXi prod = rec(prod_0, prod_1);
  cout << "Here is the calculated product:\n" << prod <<endl;

  */
}