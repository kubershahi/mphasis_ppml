#include <iostream>
#include <Eigen/Dense>
#include "read_data.hpp"

using namespace std;
using namespace Eigen;

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
void share(MatrixXf A, MatrixXf shares[])
{
  MatrixXf A_0 = MatrixXf::Random(A.rows(),A.cols());
  shares[0] = A_0;
  shares[1] = A - A_0;
}

// For integer inputs
MatrixXi rec(MatrixXi A, MatrixXi B)
{
	return A + B;
}

// For floating point inputs
MatrixXf rec(MatrixXf A, MatrixXf B)
{
  return A + B;
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
MatrixXf mult(int i, MatrixXf A, MatrixXf B, MatrixXf E, MatrixXf F, MatrixXf Z)
{ 
  //MatrixXf pp = E*F + U*F + E*V + Z;
  //MatrixXf pp = -E*F + X*F + E*w + Z;
  //MatrixXf pp = E*F + X*F + E*w + Z; //--> doesn't work (SecureML)
  MatrixXf product = MatrixXf::Random(Z.rows(),Z.cols());
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
MatrixXf linearRegression(MatrixXf X, MatrixXf Y, MatrixXf w)
{ 
  // ===========================
  // Additive Secret Sharing
  // ===========================
  MatrixXf shares[2];

  share(X, shares); // training data shares
  MatrixXf X_0 = shares[0];
  MatrixXf X_1 = shares[1];
  share(Y, shares); // label shares
  MatrixXf Y_0 = shares[0];
  MatrixXf Y_1 = shares[1];
  share(w, shares); // weight shares
  MatrixXf w_0 = shares[0];
  MatrixXf w_1 = shares[1];
  // ===========================
  // Triplet Generation (Offline Phase)
  // ===========================
  MatrixXf U = MatrixXf::Random(X.rows(),X.cols()); // masks X_i
  share(U, shares);
  MatrixXf U_0 = shares[0]; 
  MatrixXf U_1 = shares[1];

  MatrixXf E_0 = X_0 - U_0;
  MatrixXf E_1 = X_1 - U_1;
  MatrixXf E = rec(E_0, E_1); // masked X_i

  int t = (N)/B; // E = 1

  MatrixXf V = MatrixXf::Random(d, t); // masks w_i
  share(V, shares);
  MatrixXf V_0 = shares[0]; 
  MatrixXf V_1 = shares[1];

  //MatrixXf Z = U * V; // third triplet for multiplication -> change for batchwise SGD
  MatrixXf Z = MatrixXf::Zero(B,t); 
  for (int z = 0; z < Z.cols(); z++)
  {
    Z.col(z) = U.block(z * B,0,B,U.cols()) * V.col(z);
  }
  share(Z, shares);
  MatrixXf Z_0 = shares[0];
  MatrixXf Z_1 = shares[1];

  MatrixXf VV = MatrixXf::Random(B,t); // masks D_i
  share(VV, shares);
  MatrixXf VV_0 = shares[0]; 
  MatrixXf VV_1 = shares[1];

  //MatrixXf ZZ = U.transpose() * VV; // third triplet for multiplication -> change for batchwise SGD
  MatrixXf ZZ = MatrixXf::Zero(d,t);
  for (int z = 0; z < ZZ.cols(); z++)
  {
    ZZ.col(z) = U.transpose().block(0,z * B,U.cols(),B) * VV.col(z);
  }
  share(ZZ, shares);
  MatrixXf ZZ_0 = shares[0];
  MatrixXf ZZ_1 = shares[1];

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
      MatrixXf F_0 = w_0 - V_0.col(j);
      MatrixXf F_1 = w_1 - V_1.col(j);
      MatrixXf F = rec(F_0, F_1);

      // YY = X_B_j.w
      MatrixXf YY_0 = mult(0, X_0.block(B * j,0,B,X.cols()), w_0, E.block(B * j,0,B,X.cols()), F, Z_0.col(j));
      MatrixXf YY_1 = mult(1, X_1.block(B * j,0,B,X.cols()), w_1, E.block(B * j,0,B,X.cols()), F, Z_1.col(j));

      // D = X.w - Y
      MatrixXf D_0 = YY_0 - Y_0.block(B * j,0,B,Y.cols());
      MatrixXf D_1 = YY_1 - Y_1.block(B * j,0,B,Y.cols());

      // Loss Computation (for testing only)
      MatrixXf D = rec(D_0, D_1);
      MatrixXf loss = D.transpose() * D;

      MatrixXf FF_0 = D_0 - VV_0.col(j);
      MatrixXf FF_1 = D_1 - VV_1.col(j);
      MatrixXf FF = rec(FF_0, FF_1);

      // delta = X^T(X.w - Y)
      MatrixXf delta_0 = mult(0, X_0.transpose().block(0,B * j,X.cols(),B), D_0, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_0.col(j));
      MatrixXf delta_1 = mult(1, X_1.transpose().block(0,B * j,X.cols(),B), D_1, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_1.col(j));

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
MatrixXf idealLinearRegression(MatrixXf X, MatrixXf Y, MatrixXf w) // ideal functionality
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
      MatrixXf YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
      MatrixXf D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
      // Loss Computation
      MatrixXf loss = D.transpose() * D;
      MatrixXf delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
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
    w = w - (delta / (B * 100)); // w -= a/B * delta
  }
  return w;
}

MatrixXf predict(MatrixXf X, MatrixXf Y, MatrixXf w)
{ MatrixXf pred = X * w;
  MatrixXf diff = pred - Y;
  MatrixXf loss = diff.transpose() * diff;
  cout<< "Test Loss: "<< loss(0,0)/X.rows() << endl;
  return pred;
}

MatrixXf predict(MatrixXf X, MatrixXf w)
{ return X * w;}

int main(){

  //==========================================
  // MNIST data:
  //==========================================
  cout<<"Reading Data:"<<endl;
  // loading mnist dataset: training and testing portion separately
  vector<vector<float> > X_train;   // dim: 60000 x 784, 60000 training samples with 784 features
  vector<float> Y_train;            // dim: 60000 x 1  , the true label of each training sample
  read_data("datasets/mnist/mnist_train.csv", X_train, Y_train);
  
  MatrixXf X1(N, d); // 60000, 784
  MatrixXf Y1(N, 1); // 60000, 1

  for (int i = 0; i < N; i++)
  {
    X1.row(i) = VectorXf::Map(&X_train[i][0], d)/255.0; // VectorXf::Map(&X_train[i][0],X_train[i].size());
    Y1.row(i) = VectorXf::Map(&Y_train[i],1)/10.0;
  }

  vector<vector<float> > X_test;    // dim: 10000 x 784, 10000 testing samples with 784 features
  vector<float> Y_test;             // dim: 10000 x 1  , the true label of each testing sample
  read_data("datasets/mnist/mnist_test.csv", X_test, Y_test);

  MatrixXf X1_test(N_test, d); // 1000, 784
  MatrixXf Y1_test(N_test, 1); // 1000, 1

  for (int i = 0; i < N_test; i++)
  {
    X1_test.row(i) = VectorXf::Map(&X_test[i][0], d)/255.0;
    Y1_test.row(i) = VectorXf::Map(&Y_test[i],1)/10.0;
  }

  MatrixXf w1 = MatrixXf::Random(d,1);
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
  MatrixXf X2(6,2);
  X2 << 4,1,2,8,1,0,3,2,1,4,6,7;
  MatrixXf Y2(6,1);
  Y2 << 2,-14,1,-1,-7,-8;
  MatrixXf w2(2,1);
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
  MatrixXf ideal_w = idealLinearRegression(X1,Y1,w1);
  //cout << "Final weights (under Ideal Functionality) are:\n" << ideal_w <<endl;
  cout << endl << "==========================================="<<endl;
  cout << "PRIVACY-PRESERVING LINEAR REGRESSION (SGD):"<<endl;
  cout << "==========================================="<<endl<<endl;
  MatrixXf new_w = linearRegression(X1,Y1,w1);
  //cout << "Final weights (under Privacy Preserving) are:\n" << new_w <<endl;



  //==========================================
  // MODEL PREDICTION:
  //==========================================
  
  cout << endl << "==================================="<<endl;
  cout << "PREDICTION (using trained weights):"<<endl;
  cout << "==================================="<<endl<<endl;
  MatrixXf pred = predict(X1_test, Y1_test, new_w);
  //cout << pred <<endl;

  cout << endl << "Single example predictions: " << endl;

  for (int k = 123; k < 600; k += 100){
    cout << "True Label: " << Y1_test.row(k) << endl;
    MatrixXf pred_i = predict(X1_test.row(k), ideal_w);
    cout << "Ideal Prediction: " << pred_i <<endl;
    MatrixXf pred_p = predict(X1_test.row(k), new_w);
    cout << "PP Prediction: " << pred_p <<endl;
  }

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


