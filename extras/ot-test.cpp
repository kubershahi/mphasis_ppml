#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>

using namespace std;
using namespace Eigen;
using namespace emp;

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;

IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]", "[", "]");

template<class Derived>
void send(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X){
  io->send_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t));
  return;
}

template<class Derived>
void recv(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X){
  io->recv_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t));
  return;
}

// For 64-integer inputs
void share(MatrixXi64 A, MatrixXi64 shares[])
{
	MatrixXi64 A_0 = MatrixXi64::Random(A.rows(),A.cols()) / 10000000;
	shares[0] = A_0;
	shares[1] = A - A_0;
}

// For 64-integer inputs
MatrixXi64 rec(MatrixXi64 A, MatrixXi64 B)
{
  return A + B;
}

int main(int argc, char** argv)
{
  int N = 640;
  int d = 784;
  int B = 128;
  int t = N/B;

  MatrixXi64 shares[2];
  MatrixXi64 U = MatrixXi64::Random(N,d);
  share(U,shares);

  MatrixXi64 U_0 = shares[0]; 
  MatrixXi64 U_1 = shares[1];

  MatrixXi64 V = MatrixXi64::Random(d,t);
  share(V,shares);

  MatrixXi64 V_0 = shares[0];
  MatrixXi64 V_1 = shares[1];

  MatrixXi64 Z = MatrixXi64::Zero(B,t);
  MatrixXi64 Z_0 = MatrixXi64::Zero(B,t);
  MatrixXi64 Z_1 = MatrixXi64::Zero(B,t);

  for (int i = 0; i < Z.cols(); i++){
    Z.col(i) = U.block(i*B, 0, B, U.cols()) * V.col(i);
    Z_0.col(i) = U_0.block(i*B, 0, B, U_0.cols()) * V_0.col(i) + U_0.block(i*B, 0, B, U_0.cols()) * V_1.col(i);
    Z_1.col(i) = U_1.block(i*B, 0, B, U_1.cols()) * V_1.col(i) + U_1.block(i*B, 0, B, U_1.cols()) * V_0.col(i);
  }

  MatrixXi64 Z_test = rec(Z_0,Z_1);

  if (Z==Z_test){
    cout << "Matches" << endl;
  }
  else{
    cout << "Doesn't Match" << endl;
  }
  // cout << Z.format(CleanFmt)<< endl;

  cout << endl;
  cout << "========Triplet Generation Using OTs========" << endl;

  MatrixXi64 Z_0_ot = MatrixXi64::Zero(B,t);
  MatrixXi64 Z_1_ot = MatrixXi64::Zero(B,t);

  emp::PRG prg;

  MatrixXi64 Ai(N, d);
  MatrixXi64 Bi(d, t);

  prg.random_data(Ai.data(), N * d *8);

  // cout << endl;
  // cout << Ai.format(CleanFmt) << endl;

  return 0;
}