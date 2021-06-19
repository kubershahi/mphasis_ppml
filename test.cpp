#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// void insert(vector<int>  &l, int item){
//   l.push_back(item);
// }

void bin(long n)
  {
    long i;
    cout << "0";
    for (i = 1 << 30; i > 0; i = i / 2)
    {
      if((n & i) != 0)
      {
        cout << "1";
      }
      else
      {
        cout << "0";
      }
    }
  }

int main(){

  // MatrixXf X = MatrixXf::Random(1);
  // MatrixXf Y = MatrixXf::Random(1);

  double X = 10.1578;
  double Y = 5.2345;
  double Z = X * Y; // the value we want ot approximate
  

  cout << endl << "X, 1010.001010: "<< X << endl;
  cout << "Y, 101.001111: "<< Y << endl;
  cout << "Z, 110101.001010111: " << Z << endl << endl;

  X = X * 64; // precision of 6 binary decimal digits i.e 2^6
  Y = Y * 64; // precision of 6 binary decimal digits i.e 2^6

  cout << fixed;
  cout << "After scaling by 6 bits i.e 2^6: " << endl;
  cout << "X: "<< X << endl;
  cout << "Y: "<< Y << endl << endl;
  
  cout << "After changing to integer " << endl;
  int x = (int) X ;
  int y = (int) Y ;

  cout << "x, 1010001010: "<< x << endl;
  cout << "y, 101001111 : "<< y << endl << endl;
  
  int z = x * y;
  cout << "z before truncation, 110101001010010110: "<< z << endl;
  z /= 64; // truncation product
  cout << "z after  truncation, 110101001010      : "<< z << endl << endl;

  double zz = (double) z;
  zz /= 64;
  cout << "z after cast,        110101.001010     : "<< zz << endl << endl;


  // // selecting certain columns of a Matrix
  // Map<MatrixXf> X1(X.data()+5,5,X.cols());

  // casting a matrix;
  // MatrixXi x = X.cast<int>();
}