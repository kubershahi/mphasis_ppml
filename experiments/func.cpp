#include<iostream>
#include <string>
#include <cmath>

#define SCALING_FACTOR 65536 // Precision of 16 bits

using namespace std;


// funciton to convert a number from double-precision floats to 64-bit unsigned integer
uint64_t floattouint64(double a)
{
  uint64_t res;
  if ( a >= 0)
  {
    res = (uint64_t) (a * SCALING_FACTOR);
    // cout<< res << " is positive"<<endl;
  }
  else
  {
    a = abs(a * SCALING_FACTOR);
    res = (uint64_t) pow(2,64) - a;
    // cout<< res << " is negative"<<endl;
  }
  return res;
}

// function that converts a number from 64-bit unsigned integer to double matrix
double uint64tofloat(uint64_t a)
{
  double res;
  if (a & (1UL << 63))
  {
    res = - ((double) pow(2,64) - a)/SCALING_FACTOR;
    //cout<< res << " is negative"<<endl;
  }
  else
  {
    res = ((double) a)/SCALING_FACTOR;
    //cout<< res << " is positive"<<endl;
  }

  return res;
}

// function that truncates an integer value by a factor

uint64_t truncate(uint64_t a, int factor)
{
  uint64_t res;

  if (a & (1UL << 63))
  {
    res = (uint64_t) pow(2,64) - ( (uint64_t)pow(2,64) - a)/factor;
    //cout<< res(i,j) << " is negative"<<endl;
  }
  else
  {
    res = a/factor;
    //cout<< res(i,j) << " is positive"<<endl;
  }

  return res;
}

int main()
{
  double X = -10.15723545348;
  double Y = 5.23423452345;

  cout << fixed;
  cout << endl << "X: " << X << endl;
  cout << "Y: " << Y << endl;

  double Z = X * Y;
  cout << endl << "Z (X * Y: floating arithmetic): " << Z << endl << endl;

  uint64_t X_i = floattouint64(X); // mapping XX to integer
  uint64_t Y_i = floattouint64(Y); // mapping YY to integer

  // no secret sharing setting
  cout << "=== No Secret Sharing Setting (just mapping, multiplying, truncation, and reverse mapping) ==="<<endl << endl;

  uint64_t Z_i = X_i * Y_i;      // multiplying X and Y
  uint64_t Z_t = truncate(Z_i, SCALING_FACTOR); // truncating Z
  // cout << "Z truncated (unshared setting): " << Z_t << endl;
  double Z_f = uint64tofloat(Z_t);                 // mapping Z back to double

  cout << "Z (unshared setting): " << Z_f << endl << endl;

  return 0;

}