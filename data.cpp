#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "read_data.hpp"

using namespace std;

/*  dummy program to read data and test read_data function

*/
int main() {

  vector<vector<float> > X;
  vector<float > Y;
  // read_data("datasets/medical_cost/insurance.csv", X, Y);
  read_data("datasets/mnist/mnist_test.csv", X, Y);

  // cout << X[1][0]<< " " << X[1][1] << " " << X[1][2] << " " <<X[1][3] << " " << X[1][4] << endl;

  //splitting dataset 
  // int size = 10;
  // vector<vector<float> > X(X.begin(), X.begin() + size);
  // cout << X.size() << endl;
  // cout << X[1][0]<< " " << X[1][1] << " " << X[1][2] << " " <<X[1][3] << " " << X[1][4] << endl;

  // for (int i = 0; i <X.size(); i++){
  //   cout << i << ": " << X[0][i] << endl;
  // }


  for (int j = 0; j <100; j++){
    cout << j << ": " << Y[j] << " ";
  }
  cout << "\n";
  cout << X[410][773] << endl;
  cout << X.size() << endl;
  cout << Y.size() << endl;
}