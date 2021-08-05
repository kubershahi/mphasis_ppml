#include "linear_regression.hpp"
#include "utils.hpp"
#include "read_data.hpp"

#include <iostream>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

// ================================================== 
// NON-PRIVACY-PRESERVING LINEAR REGRESSION (IDEAL)
// ==================================================

// Non-PP Linear Regression for integer inputs
MatrixXi idealLinearRegression(MatrixXi X, MatrixXi Y, MatrixXi w)
{
  // w -= a/|B| X^T .(X.w - Y)
  for (int e= 0; e< NUM_EPOCHS; e++)
  { cout<< "Epoch Number: "<< e+1;
    float epoch_loss = 0.0;

    for(int i = 0; i < (N/B); i ++)
    {
      MatrixXi YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
      MatrixXi D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i

      MatrixXi loss = D.transpose() * D;

      MatrixXi delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      w = w - (delta / (B*100)); // w -= alpha/B * delta

      epoch_loss += loss(0,0);
    }
  cout<<endl;
  cout<< "Loss: "<< epoch_loss/N << endl;
  }
  return w;
}

// Non-PP Linear Regression for floating point inputs
MatrixXd idealLinearRegression(MatrixXd X, MatrixXd Y, MatrixXd w)
{
  //cout<<"Initial weights: "<< endl << w <<endl;
  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1;
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++)
    { 
      MatrixXd YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w
      //cout<< "yhat: "<< endl << YY << endl;
      MatrixXd D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
      //cout<< "diff: "<< endl << D << endl;
      // Loss Computation
      MatrixXd loss = D.transpose() * D;
      MatrixXd delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      //cout<< "grad: " << endl << delta << endl;
      w = w - (delta / (B*100)); // w -= alpha/B * delta
      //cout<<"weights: "<< endl << w <<endl;
      //cout<<w<<endl;
      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  
  return w;
}

// Non-PP Linear Regression for 64-Integer inputs
MatrixXi64 idealLinearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w) 
{
  // w -= a/|B| X^T .(X.w - Y)
  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1;
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++)
    { 
      // forward propagation
      MatrixXi64 YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w

      //truncation
      YY = truncate(YY, SCALING_FACTOR);

      // Compute difference
      MatrixXi64 D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
      
      // Loss Computation
      MatrixXd loss = uint64tofloat(D).transpose() * uint64tofloat(D);

      // Computing change in weight
      MatrixXi64 delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      //cout<< "grad_raw: " << endl << delta << endl;

      //truncation:
      delta = truncate(delta, SCALING_FACTOR);
      delta = truncate(delta, (B*100)); // alpha/B * delta,

      // weight update
      w = w - delta; // w -= alpha/B * delta
      //cout<<"weights: "<< endl << uint64tofloat(w) <<endl;

      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  
  return w;
}

// ==================================== 
// PRIVACY-PRESERVING LINEAR REGRESSION
// ==================================== 

// Linear Regression for 64-Integer Input 
MatrixXi64 linearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w)
{ 
  // ===========================
  // Additive Secret Sharing
  // ===========================
  MatrixXi64 shares[2];

  share(X, shares); // training data shares
  MatrixXi64 X_0 = shares[0];
  MatrixXi64 X_1 = shares[1];
  share(Y, shares); // label shares
  MatrixXi64 Y_0 = shares[0];
  MatrixXi64 Y_1 = shares[1];
  share(w, shares); // weight shares
  MatrixXi64 w_0 = shares[0];
  MatrixXi64 w_1 = shares[1];

  // ===========================
  // Triplet Generation (Offline Phase)
  // ===========================
  MatrixXi64 U = MatrixXi64::Random(X.rows(),X.cols()); // masks X_i
  share(U, shares);
  MatrixXi64 U_0 = shares[0]; 
  MatrixXi64 U_1 = shares[1];

  MatrixXi64 E_0 = X_0 - U_0;
  MatrixXi64 E_1 = X_1 - U_1;
  MatrixXi64 E = rec(E_0, E_1); // masked X_i

  int t = (N)/B; // E = 1

  MatrixXi64 V = MatrixXi64::Random(d, t); // masks w_i
  share(V, shares);
  MatrixXi64 V_0 = shares[0]; 
  MatrixXi64 V_1 = shares[1];

  //MatrixXi64 Z = U * V; // third triplet for multiplication -> change for batchwise SGD
  MatrixXi64 Z = MatrixXi64::Zero(B,t); 
  for (int z = 0; z < Z.cols(); z++)
  {
    Z.col(z) = U.block(z * B,0,B,U.cols()) * V.col(z);
  }
  share(Z, shares);
  MatrixXi64 Z_0 = shares[0];
  MatrixXi64 Z_1 = shares[1];

  MatrixXi64 VV = MatrixXi64::Random(B,t); // masks D_i
  share(VV, shares);
  MatrixXi64 VV_0 = shares[0]; 
  MatrixXi64 VV_1 = shares[1];

  //MatrixXi64 ZZ(Z') = U.transpose() * VV(V'); // third triplet for multiplication -> change for batchwise SGD
  MatrixXi64 ZZ = MatrixXi64::Zero(d,t);
  for (int z = 0; z < ZZ.cols(); z++)
  {
    ZZ.col(z) = U.transpose().block(0,z * B,U.cols(),B) * VV.col(z);
  }
  share(ZZ, shares);
  MatrixXi64 ZZ_0 = shares[0];
  MatrixXi64 ZZ_1 = shares[1];

  // ===========================
  // Online Phase
  // ===========================

  for(int e = 0; e < NUM_EPOCHS; e++)
  { cout<< "Epoch Number: "<< e+1;
    float epoch_loss = 0.0;

    for(int j = 0; j < int(N/B); j++)
    {
      MatrixXi64 F_0 = w_0 - V_0.col(j);
      MatrixXi64 F_1 = w_1 - V_1.col(j);
      MatrixXi64 F = rec(F_0, F_1);

      // YY = X_B_j * w  forward propagation
      MatrixXi64 YY_0 = mult(0, X_0.block(B * j,0,B,X.cols()), w_0, E.block(B * j,0,B,X.cols()), F, Z_0.col(j));
      MatrixXi64 YY_1 = mult(1, X_1.block(B * j,0,B,X.cols()), w_1, E.block(B * j,0,B,X.cols()), F, Z_1.col(j));
      // truncation
      YY_0 = truncate(YY_0, SCALING_FACTOR);
      YY_1 = truncate(YY_1, SCALING_FACTOR);

      // D = X.w - Y compute difference
      MatrixXi64 D_0 = YY_0 - Y_0.block(B * j,0,B,Y.cols());
      MatrixXi64 D_1 = YY_1 - Y_1.block(B * j,0,B,Y.cols());

      // Loss Computation (for testing only)
      MatrixXi64 D = rec(D_0, D_1);
      MatrixXd loss = uint64tofloat(D).transpose() * uint64tofloat(D);

      MatrixXi64 FF_0 = D_0 - VV_0.col(j);
      MatrixXi64 FF_1 = D_1 - VV_1.col(j);
      MatrixXi64 FF = rec(FF_0, FF_1);

      // delta = X^T(X.w - Y)  computing change in weight
      MatrixXi64 delta_0 = mult(0, X_0.transpose().block(0,B * j,X.cols(),B), D_0, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_0.col(j));
      MatrixXi64 delta_1 = mult(1, X_1.transpose().block(0,B * j,X.cols(),B), D_1, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_1.col(j));
      
      // truncation
      delta_0 = truncate(delta_0, SCALING_FACTOR);
      delta_1 = truncate(delta_1, SCALING_FACTOR);
      
      // gradient update
      delta_0 = truncate(delta_0, (B*100)); // alpha/B * delta, 
      delta_1 = truncate(delta_1, (B*100)); // alpha/B * delta, 
      w_0 = w_0 - delta_0; // w = w - alpha/B * delta
      w_1 = w_1 - delta_1; // w = w - alpha/B * delta

      //test
      MatrixXi64 w = rec(w_0, w_1);
      //cout<<"weights: "<< endl << uint64tofloat(w) <<endl;

      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
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

  //MatrixXd ZZ(Z') = U.transpose() * VV(V'); // third triplet for multiplication -> change for batchwise SGD
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
    float epoch_loss = 0.0;

    for(int j = 0; j < int(N/B); j++)
    {
      MatrixXd F_0 = w_0 - V_0.col(j);
      MatrixXd F_1 = w_1 - V_1.col(j);
      MatrixXd F = rec(F_0, F_1);

      // YY = X_B_j.w forward propagation
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

      // delta = X^T(X.w - Y) computing change in weight
      MatrixXd delta_0 = mult(0, X_0.transpose().block(0,B * j,X.cols(),B), D_0, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_0.col(j));
      MatrixXd delta_1 = mult(1, X_1.transpose().block(0,B * j,X.cols(),B), D_1, E.transpose().block(0,B * j,X.cols(),B), FF, ZZ_1.col(j));

      // delta = X^T(X.w - Y) updating weight
      w_0 = w_0 - (delta_0 / (B*100)); // w = w - alpha/B * delta
      w_1 = w_1 - (delta_1 / (B*100)); // w = w - alpha/B * delta

      //cout<< rec(w_0, w_1) <<endl;
      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  
  return rec(w_0, w_1);
}