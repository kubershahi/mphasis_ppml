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
  vector<vector<float> > dataset = read_data("datasets/mnist/mnist_test.csv");

  // for (int j = 0; j < 10; j++ ){
  //   cout << dataset[j][0]<< " " << dataset[j][1] << " " << dataset[j][2] << " " <<dataset[j][3] << " " << dataset[j][4] << " " << dataset[j][5] << endl;
  // }

//splittin dataset
  // int size = 10;
  // vector<vector<float> > train_X(dataset.begin(), dataset.begin() + size);
  // cout << train_X.size() << endl;
  // cout << train_X[1][0]<< " " << train_X[1][1] << " " << train_X[1][2] << " " <<train_X[1][3] << " " << train_X[1][4] << " " << train_X[1][5] << endl;

  

  for (int i = 0; i <785; i++){
    cout << i << ": " << dataset[0][i] << endl;
  }

  cout << dataset.size() << endl;
}