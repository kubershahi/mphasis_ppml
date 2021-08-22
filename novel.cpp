#include <iostream>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

// Given U, find f(U) privately, where f = Double ReLU

// f(z) = 0, if z < -1/2
// f(z) = z + 1/2, if -1/2 < z < 1/2
// f(z) = 1, if z > 1/2

// For integer inputs
MatrixXi mult(int i, MatrixXi A, MatrixXi B, MatrixXi E, MatrixXi F, MatrixXi Z)
{ 
  MatrixXi product = MatrixXi::Random(Z.rows(),Z.cols());
  if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
  else if (i == 0) product = (A * F) + (E * B) + Z;
  
  return product;
}

// For 64-integer inputs
MatrixXi64 mult(int i, MatrixXi64 A, MatrixXi64 B, MatrixXi64 E, MatrixXi64 F, MatrixXi64 Z)
{ 
	MatrixXi64 product = MatrixXi64::Random(Z.rows(),Z.cols());
	if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
	else if (i == 0) product = (A * F) + (E * B) + Z;
	
	return product;
}



int main(){
	double z_0 = 0.98;
	double z_1 = 0.54;
	double theta_0 = z_0 + 0.25;
	double theta_1 = z_1 + 0.25;
	double mult_fact_0 = 2.32;
	double mult_fact_0 = 1.92;
	double r_0 = theta_0 * mult_fact;
	double r_1 = theta_1 * mult_fact;
	double theta = 



	return 0;
}


