#include "logistic_regression.hpp"

#include <iostream>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;


// ================================================== 
// Activation Function and Helpers
// ================================================== 

// Given U, find f(U) privately, where f = Double ReLU

// f(z) = 0, if z < -1/2
// f(z) = z + 1/2, if -1/2 < z < 1/2
// f(z) = 1, if z > 1/2

// ===========================
// Helper Functions:
// ===========================

// function that converts float to int
uint64_t floattouint64(double d)
{ uint64_t res;
  if (d > 0) res = (uint64_t)(d * SCALING_FACTOR);
  else res = (uint64_t) pow(2,64) - abs(d * SCALING_FACTOR);
  return res;
}

double uint64tofloat(uint64_t a)
{
  double res;
  if (a & (1UL << 63)) res = -((double) pow(2,64) - a)/((double)SCALING_FACTOR); // negative
  else res = ((double) a)/((double)SCALING_FACTOR);
  return res;
}

uint64_t truncate(uint64_t a, int factor)
{
  uint64_t res;
  if (a & (1UL << 63)) res = (uint64_t) pow(2,64) - ( (uint64_t)pow(2,64) - a)/factor;
  else res = a/factor;
  return res;
}

// For floating inputs
double mult(double i, double A, double B, double E, double F, double Z)
{ 
  double product = 0;
  if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
  else if (i == 0) product = (A * F) + (E * B) + Z;
  
  return product;
}

// For integer inputs
uint64_t mult(uint64_t i, uint64_t A, uint64_t B, uint64_t E, uint64_t F, uint64_t Z)
{ 
  uint64_t product = 0;
  if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z*SCALING_FACTOR;
  else if (i == 0) product = (A * F) + (E * B) + Z*SCALING_FACTOR;
  
  return product;
}

// For floating inputs
void share(double A, double shares[])
{ 
  double A_0 = ((double)rand()) / ((double)RAND_MAX) * A + 0.1;
  shares[0] = A_0;
  shares[1] = A - A_0;
}

// For integer inputs
void share(uint64_t A, uint64_t shares[])
{ 
  uint64_t A_0 = (uint64_t) (rand() % A);
  shares[0] = A_0;
  shares[1] = A - A_0;
}

// For floating inputs
double rec(double A, double B)
{ 
  return A + B;
}

double activation(double theta){
  //====
  double U = 2.21;
  double V = 1.99;
  double Z = U * V;
  //===
  //test(U,V,Z);
  double fout;

  double delta = theta + 0.5;
  double gamma = theta - 0.5;
  double delta_shares[2];
  double gamma_shares[2];
  share(delta, delta_shares);
  share(gamma, gamma_shares);
  //cout<< "delta: "<< delta <<" = "<< delta_shares[0] <<" + "<< delta_shares[1] << endl;

  double r = 0.4131; // to be tested
  double r_shares[2]; // change this
  share(r, r_shares);
  //cout<< "r: "<< r <<" = "<< r_shares[0] <<" + "<< r_shares[1] << endl;

  double E = r - U;
  double F = delta - V;
  double Z_shares[2];
  share(Z, Z_shares);
  // Party 0
  double rdelta_0 = mult(0, r_shares[0], delta_shares[0], E, F, Z_shares[0]);
  // Party 1
  double rdelta_1 = mult(1, r_shares[1], delta_shares[1], E, F, Z_shares[1]);

  double rdelta = rec(rdelta_0, rdelta_1);
  //cout<< "rdelta: "<< rdelta <<" = "<< rdelta_0 <<" + "<< rdelta_1 << endl;
  //cout<<"theta (floating): "<<theta<<", delta: "<<delta<<", rdelta: "<< rdelta <<endl;

  // ====================
  uint64_t thetaint = floattouint64(theta);
  uint64_t deltaint = floattouint64(delta);
  uint64_t gammaint = floattouint64(gamma);
  uint64_t deltaint_shares[2];
  deltaint_shares[0] = floattouint64(delta_shares[0]);
  deltaint_shares[1] = floattouint64(delta_shares[1]);
  uint64_t gammaint_shares[2];
  gammaint_shares[0] = floattouint64(gamma_shares[0]);
  gammaint_shares[1] = floattouint64(gamma_shares[1]);

  uint64_t rint = floattouint64(r);
  uint64_t rint_shares[2];
  rint_shares[0] = floattouint64(r_shares[0]);
  rint_shares[1] = floattouint64(r_shares[1]);

  uint64_t Zint_shares[2];
  //share(Zint, Zint_shares);
  Zint_shares[0] = floattouint64(Z_shares[0]);
  Zint_shares[1] = floattouint64(Z_shares[1]);
  //uint64_t Eint = num1int - Uint;
  //uint64_t Fint = num2int - Vint;
  uint64_t Eint = floattouint64(E);
  uint64_t Fint = floattouint64(F);
  // Party 0
  uint64_t rdeltaint_0 = mult(0, rint_shares[0], deltaint_shares[0], Eint, Fint, Zint_shares[0]);
  // Party 1
  uint64_t rdeltaint_1 = mult(1, rint_shares[1], deltaint_shares[1], Eint, Fint, Zint_shares[1]);
  
  rdeltaint_0 = truncate(rdeltaint_0, SCALING_FACTOR);
  rdeltaint_1 = truncate(rdeltaint_1, SCALING_FACTOR);

  //cout<< "theta (uint): "<< uint64tofloat(thetaint) <<", delta: "<<uint64tofloat(deltaint)<<", rdelta: "<< uint64tofloat(rdeltaint_0 + rdeltaint_1) <<endl;
  if (rdelta < 0) fout = 0;
  else
  {
    double s = 3.0; 
    double s_shares[2];
    share(s, s_shares);
    //cout<< "s: "<< s <<" = "<< s_shares[0] <<" + "<< s_shares[1] << endl;

    E = s - U;
    F = gamma - V;
    // Party 0
    double sgamma_0 = mult(0, s_shares[0], gamma_shares[0], E, F, Z_shares[0]);
    // Party 1
    double sgamma_1 = mult(1, s_shares[1], gamma_shares[1], E, F, Z_shares[1]);
    
    double sgamma = rec(sgamma_0, sgamma_1);
    //cout<< "sgamma: "<< sgamma <<" = "<< sgamma_0 <<" + "<< sgamma_1 << endl;
    //cout<<"theta: "<<theta<<", gamma: "<<gamma<<", sgamma: "<< sgamma <<endl;
    if (sgamma < 0) fout = theta + 0.5;
    else fout = 1;

  }
  return fout;
}

// ================================================== 


// ================================================== 
// NON-PRIVACY-PRESERVING LOGISTIC REGRESSION (IDEAL)
// ==================================================

// Non-PP Logistic Regression for floating point inputs
MatrixXd idealLogisticRegression(MatrixXd X, MatrixXd Y, MatrixXd w)
{
  //cout<<"Initial weights: "<< endl << w <<endl;
  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1;
    float epoch_loss = 0.0;

      for(int i = 0; i < int(N/B); i ++)
      { 
      MatrixXd YY = X.block(B * i,0,B,X.cols()) * w;  // YY = X_B_i.w
      //cout<<YY(0,0)<<endl;
      YY = YY.unaryExpr(&sigmoid); // applying sigmoid activation function
      //cout<<YY(0,0)<<endl;
      //cout<<endl;                    
      // cout<< "yhat: "<< endl << YY << endl;

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

// ==================================== 
// PRIVACY-PRESERVING LOGISTIC REGRESSION
// ==================================== 

// Logistic Regression for 64-Integer Input 
MatrixXi64 logisticRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w)
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
      //cout<<"Before activation:"<<uint64tofloat(YY_0(0,0) + YY_1(0,0)) <<endl;

      // Add activation function here
      // ===========================
      for (int ii = 0; ii < YY_0.rows(); ii++){
        for (int jj = 0; jj < YY_0.cols(); jj++){
          //cout<<"Before activation (clear):"<<uint64tofloat(YY_0(ii, jj) + YY_1(ii, jj))<<endl;
          double theta = uint64tofloat(YY_0(ii, jj) + YY_1(ii, jj));
          double out = activation(theta);
          //cout<<"After activation (clear):"<< out <<endl;
          double out_shares[2];
          share(out, out_shares);
          YY_0(ii, jj) = floattouint64(out_shares[0]);
          YY_1(ii, jj) = floattouint64(out_shares[1]);
          //cout<<"After activation (clear + rec):"<< uint64tofloat(YY_0(ii, jj) + YY_1(ii, jj)) <<endl;
        }
      }
      //cout<<"After activation (rec):"<< uint64tofloat(YY_0(0,0) + YY_1(0,0)) <<endl<<endl;
      // ===========================

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