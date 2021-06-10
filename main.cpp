#include <iostream>
#include <Eigen/Dense>
 
using namespace std;
using namespace Eigen;

/*
====================================
Goals:
- Additive Secret Sharing [done]
- Adding shares [done]
- Multiplying shares using triplets [done]
- Online Phase of Linear Regression [done]
- Ideal Functionality (Linear Regression) [done]
- Batchwise SGD []
- Use NetIO for socket communication []
- Truncation (later)
====================================
Parameters:
- n: number of samples
- d: number of features
- B: batch size
- t: n/B
====================================
Dimensions:
- X: (n,d)
- Y: (n,1)
- U: (n,d)
- V: (d,t)
- V': (B,t)
- Z: (B,t)
- Z': ()
====================================
*/
 
void share(MatrixXi A, MatrixXi shares[])
{
	MatrixXi A_0 = MatrixXi::Random(A.rows(),A.cols()) / 10000000;
	shares[0] = A_0;
	shares[1] = A - A_0;
}

MatrixXi rec(MatrixXi A, MatrixXi B)
{
	return A + B;
}

MatrixXi mult(int i, MatrixXi A, MatrixXi B, MatrixXi E, MatrixXi F, MatrixXi Z)
{
	MatrixXi product = MatrixXi::Random(Z.rows(),Z.cols());
	if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
	else if (i == 0) product = (A * F) + (E * B) + Z;
	
	return product;
}

MatrixXi linearRegression(MatrixXi X, MatrixXi Y, MatrixXi w)
{ 
  MatrixXi shares[2];
  //MatrixXi w = MatrixXi::Random(5,1) / 10000000;
  share(w, shares);
	MatrixXi w_0 = shares[0];
  MatrixXi w_1 = shares[1];

  // Additive Secret Sharing
  // ===========================
  

  share(X, shares); // training data shares
  MatrixXi X_0 = shares[0];
  MatrixXi X_1 = shares[1];
  share(Y, shares); // label shares
  MatrixXi Y_0 = shares[0];
  MatrixXi Y_1 = shares[1];
  // ===========================

  // Masking
  // ===========================
  MatrixXi U = MatrixXi::Random(X.rows(),X.cols()) / 10000000; // masks X_i
  share(U, shares);
  MatrixXi U_0 = shares[0]; 
  MatrixXi U_1 = shares[1];

  MatrixXi E_0 = X_0 - U_0;
  MatrixXi E_1 = X_1 - U_1;
  MatrixXi E = rec(E_0, E_1);

  cout << "Here is the matrix U + E:\n" << U + E <<endl; // sanity check, must = X
  // =================================================================================
  // following to be in a for loop, for 't' epochs:

  MatrixXi V = MatrixXi::Random(w.rows(),w.cols()) / 10000000; // masks Y_i
  share(V, shares);
  MatrixXi V_0 = shares[0]; 
  MatrixXi V_1 = shares[1];

  MatrixXi F_0 = w_0 - V_0;
  MatrixXi F_1 = w_1 - V_1;
  MatrixXi F = rec(F_0, F_1);

  //cout << "Dim of U: " << U.rows()<<" "<<U.cols() <<endl;
  //cout << "Dim of V: " << V.rows()<<" "<<V.cols() <<endl;

  MatrixXi Z = U * V; // third triplet for multiplication
  share(Z, shares);
  MatrixXi Z_0 = shares[0];
  MatrixXi Z_1 = shares[1];
  // =========================== 

  // YY = X.w
  MatrixXi YY_0 = mult(0, X_0, w_0, E, F, Z_0);
  MatrixXi YY_1 = mult(1, X_1, w_1, E, F, Z_1);

  // D = X.w - Y
  MatrixXi D_0 = YY_0 - Y_0;
  MatrixXi D_1 = YY_1 - Y_1;

  // =========================== 
  MatrixXi VV = MatrixXi::Random(D_0.rows(),D_0.cols()) / 10000000; // masks D_i
  share(VV, shares);
  MatrixXi VV_0 = shares[0]; 
  MatrixXi VV_1 = shares[1];

  MatrixXi FF_0 = D_0 - VV_0;
  MatrixXi FF_1 = D_1 - VV_1;
  MatrixXi FF = rec(FF_0, FF_1);

  MatrixXi ZZ = U.transpose() * VV; // third triplet for multiplication
  share(ZZ, shares);
  MatrixXi ZZ_0 = shares[0];
  MatrixXi ZZ_1 = shares[1];
  // =========================== 

  // delta = X^T(X.w - Y)
  MatrixXi delta_0 = mult(0, X_0.transpose(), D_0, E.transpose(), FF, ZZ_0);
  MatrixXi delta_1 = mult(1, X_1.transpose(), D_1, E.transpose(), FF, ZZ_1);

  // Truncation
  // =========================== 
  // =========================== 

  // delta = X^T(X.w - Y)
  w_0 = w_0 - (delta_0 / 128); // alpha/b = 2^-7 = 128; used in paper
  w_1 = w_1 - (delta_1 / 128); // alpha/b = 2^-7 = 128; used in paper

  // =========================== end for loop here ('t' epochs)
  return rec(w_0, w_1);
}

MatrixXi idealLinearRegression(MatrixXi X, MatrixXi Y, MatrixXi w) // ideal functionality
{
  // w -= a/|B| X^T (X.w - Y)
  // =================================================================================
  // following to be in a for loop, for 't' epochs:

  MatrixXi YY = X * w; // YY = X.w
  MatrixXi D = YY - Y; // D = X.w - Y
  MatrixXi delta = X.transpose() * D; // delta = X^T(X.w - Y)
  w = w - (delta / 128); // delta = X^T(X.w - Y)

  // =========================== end for loop here ('t' epochs)
  return w;
}

int main()
{
  MatrixXi X = MatrixXi::Random(3,5) / 10000000; // n = 3, d = 5, training samples
  cout << "Here is the matrix X:\n" << X <<endl;
  MatrixXi Y = MatrixXi::Random(3,1) / 10000000; // labels
  //cout << "Here is the matrix Y:\n" << Y <<endl;
  MatrixXi w = MatrixXi::Random(5,1) / 10000000;

  MatrixXi new_w = linearRegression(X,Y,w);
  cout << "Final weights are:\n" << new_w <<endl;
  MatrixXi ideal_w = idealLinearRegression(X,Y,w);
  cout << "Final weights are:\n" << ideal_w <<endl;
}

  // Additive Secret Sharing
  /*
  MatrixXi shares[2];

  share(X, shares);
  MatrixXi X_0 = shares[0];
  MatrixXi X_1 = shares[1];
  share(Y, shares);
  MatrixXi Y_0 = shares[0];
  MatrixXi Y_1 = shares[1];
  */

  //cout << "Here is the matrix X_0 + X_1:\n" << X_0 + X_1 <<endl;
  //cout << "Here is the matrix Y_0 + Y_1:\n" << Y_0 + Y_1 <<endl;

  // Multiplication
  //MatrixXi w_0 = MatrixXi::Random(5,1) / 10000000;
  //MatrixXi w_1 = MatrixXi::Random(5,1) / 10000000;
  //MatrixXi w = rec(w_0, w_1);
  //cout << "Here is the matrix w_0 + w_1:\n" << w_0 + w_1 <<endl;
  
  //Masking
  /*
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


  //MatrixXi pp = E*F + U*F + E*V + Z;
  //MatrixXi pp = -E*F + X*F + E*w + Z;
  //MatrixXi pp = E*F + X*F + E*w + Z; //--> doesn't work (SecureML)