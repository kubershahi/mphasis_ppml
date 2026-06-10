#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include<math.h>
#define SCALING_FACTOR 65536// Precision of 16 bits 8192
using namespace std;

// Given U, find f(U) privately, where f = Double ReLU

// f(z) = 0, if z < -1/2
// f(z) = z + 1/2, if -1/2 < z < 1/2
// f(z) = 1, if z > 1/2

// Helper Functions:
// ===========================

// function that converts float to int
uint64_t floattouint64(double d)
{	uint64_t res;
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
  //cout<<"Double Mult Unit Test(1): = "<< -(E * F) + (A * F) + (E * B) + Z <<endl;
  //cout<<"Float Mult Unit Test(0): "<< (A * F) + (E * B) + Z <<endl;
  //cout<<"Float Mult Unit Test(A*F): "<< (A * F) <<endl;
  //cout<<"Float Mult Unit Test(E*B): "<< (E * B) <<endl;
  //cout<<"Float Mult Unit Test(Z): "<< Z <<endl;
  if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
  else if (i == 0) product = (A * F) + (E * B) + Z;
  
  return product;
}

// For integer inputs
uint64_t mult(uint64_t i, uint64_t A, uint64_t B, uint64_t E, uint64_t F, uint64_t Z)
{ 
  uint64_t product = 0;
  //cout<<"UINT Mult Unit Test(0): "<< uint64tofloat(truncate(((A * F) + (E * B) + Z*SCALING_FACTOR), SCALING_FACTOR)) <<endl;
  //cout<<"UINT Mult Unit Test(A*F): "<< uint64tofloat(truncate((A * F), SCALING_FACTOR)) <<endl;
  //cout<<"UINT Mult Unit Test(E*B): "<< uint64tofloat(truncate((E * B), SCALING_FACTOR)) <<endl;
  //cout<<"UINT Mult Unit Test(Z): "<< uint64tofloat(truncate((Z*SCALING_FACTOR), SCALING_FACTOR)) <<endl;
  //cout<<"UINT Mult Unit Test(1): "<< uint64tofloat(truncate((-(E * F) + (A * F) + (E * B) + Z), SCALING_FACTOR)) <<endl;
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

void test(double U, double V, double Z){
	//cout<<"Test Float: "<<3.445<<" "<<-0.852<<endl;
	//cout<<"Test Float Mapped to UINT: "<<floattouint64(3.445)<<" "<<floattouint64(-0.852)<<endl;
	//cout<<"Test UNIT Mapped to Float: "<<uint64tofloat(floattouint64(3.445))<<" "<<uint64tofloat(floattouint64(-0.852))<<endl;

	double u = 123.4;
	double u_shares[2];
	share(u, u_shares);
	//cout<< "u: " << u <<" = "<< u_shares[0] <<" + "<< u_shares[1] << endl;

	double num1 = 0.54224;
	double num2 = -0.13241;
	double num1_shares[2];
	double num2_shares[2];
	share(num1, num1_shares);
	share(num2, num2_shares);
	
	//cout<< num1 <<" = "<< num1_shares[0] <<" + "<< num1_shares[1] << endl;
	//cout<< num2 <<" = "<< num2_shares[0] <<" + "<< num2_shares[1] << endl;

	double Z_shares[2];
	share(Z, Z_shares);
	double E = num1 - U;
	double F = num2 - V;

	// Party 0
	//cout<<"Inputs to Mult 0: "<<0<<" "<<num1_shares[0]<<" "<<num2_shares[0]<<" "<<E<<" "<<F<<" "<<Z_shares[0]<<endl;
	double p_0 = mult(0, num1_shares[0], num2_shares[0], E, F, Z_shares[0]);
	// Party 1
	//cout<<"Inputs to Mult 1: "<<1<<" "<<num1_shares[1]<<" "<<num2_shares[1]<<" "<<E<<" "<<F<<" "<<Z_shares[1]<<endl;
	double p_1 = mult(1, num1_shares[1], num2_shares[1], E, F, Z_shares[1]);

	cout<< "floating point: "<< num1 * num2 <<" = "<< p_0 <<" + "<< p_1 <<" = "<< p_0 + p_1 << endl;

	cout<<"========="<<endl;

	uint64_t num1int = floattouint64(num1);
	uint64_t num2int = floattouint64(num2);
	uint64_t num1int_shares[2];
	uint64_t num2int_shares[2];
	num1int_shares[0] = floattouint64(num1_shares[0]);
	num1int_shares[1] = floattouint64(num1_shares[1]);
	num2int_shares[0] = floattouint64(num2_shares[0]);
	num2int_shares[1] = floattouint64(num2_shares[1]);
	//share(num1int, num1int_shares);
	//share(num2int, num2int_shares);
	//cout<< "num1 (int64): " << uint64tofloat(num1int) <<" = "<< uint64tofloat(num1int_shares[0]) <<" + "<< uint64tofloat(num1int_shares[1]) << endl;
	//cout<< "num2 (int64): " << uint64tofloat(num2int) <<" = "<< uint64tofloat(num2int_shares[0]) <<" + "<< uint64tofloat(num2int_shares[1]) << endl;
	//===
	uint64_t Uint = floattouint64(U);
	uint64_t Vint = floattouint64(V);
	uint64_t Zint = floattouint64(Z);
	//===

	uint64_t Zint_shares[2];
	//share(Zint, Zint_shares);
	Zint_shares[0] = floattouint64(Z_shares[0]);
	Zint_shares[1] = floattouint64(Z_shares[1]);
	//uint64_t Eint = num1int - Uint;
	//uint64_t Fint = num2int - Vint;
	uint64_t Eint = floattouint64(E);
	uint64_t Fint = floattouint64(F);

	// Party 0
	//cout<<"Inputs to Mult UInt 0: "<<0<<" "<< uint64tofloat(num1int_shares[0])<<" "<< uint64tofloat(num2int_shares[0])<<" "<< uint64tofloat(Eint) <<" "<<uint64tofloat(Fint)<<" "<< uint64tofloat(Zint_shares[0])<<endl;
	uint64_t p_0int = mult(0, num1int_shares[0], num2int_shares[0], Eint, Fint, Zint_shares[0]);
	// Party 1
	//cout<<"Inputs to Mult UInt 1: "<<1<<" "<< uint64tofloat(num1int_shares[1])<<" "<< uint64tofloat(num2int_shares[1])<<" "<< uint64tofloat(Eint) <<" "<<uint64tofloat(Fint)<<" "<< uint64tofloat(Zint_shares[1])<<endl;
	uint64_t p_1int = mult(1, num1int_shares[1], num2int_shares[1], Eint, Fint, Zint_shares[1]);
	
	p_0int = truncate(p_0int, SCALING_FACTOR);
	p_1int = truncate(p_1int, SCALING_FACTOR);

	cout<< "uint64 mapped: "<< uint64tofloat(truncate(num1int * num2int, SCALING_FACTOR)) <<" = "<< uint64tofloat(p_0int)<<" + "<< uint64tofloat(p_1int) <<" = "<< uint64tofloat(p_0int + p_1int) << endl;
	cout<<"========="<<endl;

}

// ===========================

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




int main(){
	srand(time(0));
	//====
	cout<<"Enter value of theta (input to activation): ";
	double theta;
	cin>>theta;
	cout<<"Theta: "<<theta<<";   ReLU output: "<<activation(theta)<<endl;
	double out = activation(theta);
  double out_shares[2];
  share(out, out_shares);
  uint64_t num1 = floattouint64(out_shares[0]);
  uint64_t num2 = floattouint64(out_shares[1]);
  cout<<"After activation:"<< uint64tofloat(num1 + num2) <<endl<<endl;
	return 0;
}



// 0.171021
// 0.670898
